#include "gfe/gfe_dynamics.hpp"
#include "gfe/gfe_assumptions.hpp"

#include <numeric>
#include <stdexcept>

namespace gfe {

namespace {

State zeros_like(const State& x) {
    return State(x.size(), 0.0);
}

void validate_state_sizes(const State& a, const State& b, const char* msg) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(msg);
    }
}

State maybe_apply(const std::function<State(const State&, double)>& fn, const State& u, double t) {
    return fn ? fn(u, t) : zeros_like(u);
}

State maybe_force(const std::function<State(double)>& fn, const State& u, double t) {
    if (!fn) {
        return zeros_like(u);
    }
    State out = fn(t);
    validate_state_sizes(out, u, "forcing(t) must return same dimension as u");
    return out;
}

} // namespace

StepResult step_augmented(const State& u,
                          const Vec& chi,
                          double t,
                          const MemoryKernelParams& kernel,
                          const DynamicsConfig& cfg,
                          const OperatorCallbacks& ops,
                          const GSTAdapter* gst,
                          AssumptionChecks* assumption_checks,
                          EnergyObserver* energy_observer) {
    if (u.empty()) {
        throw std::invalid_argument("u must be non-empty");
    }
    if (cfg.dt <= 0.0) {
        throw std::invalid_argument("cfg.dt must be positive");
    }
    if (cfg.coupling_index >= u.size()) {
        throw std::invalid_argument("cfg.coupling_index out of range");
    }
    if (!ops.apply_L) {
        throw std::invalid_argument("ops.apply_L is required");
    }
    if (kernel.gamma.size() != kernel.w.size() || kernel.gamma.empty()) {
        throw std::invalid_argument("kernel.gamma and kernel.w must be same non-zero size");
    }
    if (chi.size() != kernel.gamma.size()) {
        throw std::invalid_argument("chi size must match memory channel count");
    }
    if (cfg.form == CouplingForm::ResonantSecondOrder) {
        if (u.size() % 2 != 0) {
            throw std::invalid_argument("ResonantSecondOrder requires even state size with [q..., v...] layout");
        }
        const std::size_t n = u.size() / 2;
        if (cfg.coupling_index >= n) {
            throw std::invalid_argument("ResonantSecondOrder coupling_index must select a q-channel (index < state_size/2)");
        }
    }

    // Non-blocking theorem-assumption evaluation.
    if (assumption_checks) {
        AssumptionOperatorCallbacks eval_ops;
        eval_ops.apply_L = ops.apply_L;
        eval_ops.apply_N = ops.apply_N;
        const auto eval = evaluate_assumptions(kernel.gamma, kernel.w, u, t, eval_ops);
        assumption_checks->pass = eval.pass;
        assumption_checks->kernel_approximable = eval.kernel_approximable;
        assumption_checks->regularity_ok = eval.regularity_ok;
        assumption_checks->operator_boundedness_ok = eval.operator_boundedness_ok;
        assumption_checks->kernel_margin = eval.kernel_margin;
        assumption_checks->regularity_margin = eval.regularity_margin;
        assumption_checks->operator_boundedness_margin = eval.operator_boundedness_margin;
        assumption_checks->note = eval.note;
    }

    const State work_u = gst ? gst->project(u, t) : u;

    State Lu = ops.apply_L(work_u, t);
    validate_state_sizes(Lu, work_u, "apply_L(u,t) must return same dimension as u");

    State Nu = maybe_apply(ops.apply_N, work_u, t);
    validate_state_sizes(Nu, work_u, "apply_N(u,t) must return same dimension as u");

    State F = maybe_force(ops.forcing, work_u, t);

    // Scalar memory feedback aggregated from channels.
    const double mem_feedback = std::inner_product(kernel.w.begin(), kernel.w.end(), chi.begin(), 0.0);

    State du(work_u.size(), 0.0);
    if (cfg.form == CouplingForm::InputDriven || cfg.form == CouplingForm::NegativeFeedback) {
        du = Lu;
        for (std::size_t i = 0; i < du.size(); ++i) {
            du[i] += Nu[i] + F[i];
        }
        if (cfg.form == CouplingForm::InputDriven) {
            du[cfg.coupling_index] += mem_feedback;
        } else {
            du[cfg.coupling_index] -= mem_feedback;
        }
    } else {
        const std::size_t n = work_u.size() / 2;
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t q_idx = i;
            const std::size_t v_idx = i + n;
            du[q_idx] = work_u[v_idx];

            double accel = Lu[v_idx] + Nu[v_idx] + F[v_idx];
            if (i == cfg.coupling_index) {
                accel -= cfg.resonant_stiffness * work_u[q_idx];
                accel -= cfg.resonant_damping * work_u[v_idx];
                accel -= cfg.resonant_feedback_gain * mem_feedback;
            }
            du[v_idx] = accel;
        }
    }

    State u_next_modal(work_u.size(), 0.0);
    for (std::size_t i = 0; i < work_u.size(); ++i) {
        u_next_modal[i] = work_u[i] + cfg.dt * du[i];
    }

    // Per formulation, drive chi with forcing or state coupling channel.
    double chi_drive = 0.0;
    if (cfg.form == CouplingForm::InputDriven) {
        chi_drive = F[cfg.coupling_index];
    } else if (cfg.form == CouplingForm::NegativeFeedback) {
        chi_drive = work_u[cfg.coupling_index];
    } else { // ResonantSecondOrder
        const std::size_t n = work_u.size() / 2;
        chi_drive = work_u[cfg.coupling_index + n];
    }

    Vec chi_next(chi.size(), 0.0);
    for (std::size_t l = 0; l < chi.size(); ++l) {
        chi_next[l] = chi[l] + cfg.dt * (-kernel.gamma[l] * chi[l] + chi_drive);
    }

    const State u_next = gst ? gst->reconstruct(u_next_modal, t + cfg.dt) : u_next_modal;

    if (energy_observer) {
        energy_observer->on_transition(t, t + cfg.dt, u, chi, u_next, chi_next);
        energy_observer->on_step(t + cfg.dt, u_next, chi_next);
    }

    return StepResult{u_next, chi_next};
}

} // namespace gfe
