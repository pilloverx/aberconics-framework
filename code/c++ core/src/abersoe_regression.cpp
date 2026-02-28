#include "abersoe/abersoe_regression.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace abersoe {
namespace {

void check_abs(const char* name,
               double observed,
               double baseline,
               double tol,
               RegressionCheckResult& out) {
    if (std::fabs(observed - baseline) > tol) {
        out.pass = false;
        out.failures.push_back(std::string(name) + " drift: observed=" + std::to_string(observed) +
                               " baseline=" + std::to_string(baseline) + " tol=" + std::to_string(tol));
    }
}

} // namespace

RegressionSnapshot make_regression_snapshot(const AberSOERunResult& run) {
    RegressionSnapshot s;
    s.state_dim = run.final_state.u.size();
    s.memory_channels = run.final_state.chi.size();
    s.steps_executed = run.diagnostics.steps_executed;
    s.all_finite = run.diagnostics.all_finite;
    s.max_abs_u = run.diagnostics.max_abs_u;
    s.max_abs_chi = run.diagnostics.max_abs_chi;
    s.final_u_l2 = run.diagnostics.final_u_l2;
    s.final_chi_l2 = run.diagnostics.final_chi_l2;
    s.mean_abs_u = run.diagnostics.mean_abs_u;
    s.mean_abs_chi = run.diagnostics.mean_abs_chi;
    return s;
}

RegressionCheckResult check_regression_snapshot(const RegressionSnapshot& observed,
                                                const RegressionSnapshot& baseline,
                                                const RegressionTolerance& tolerance) {
    RegressionCheckResult out;

    if (observed.state_dim != baseline.state_dim) {
        out.pass = false;
        out.failures.push_back("state_dim mismatch");
    }
    if (observed.memory_channels != baseline.memory_channels) {
        out.pass = false;
        out.failures.push_back("memory_channels mismatch");
    }
    if (observed.steps_executed != baseline.steps_executed) {
        out.pass = false;
        out.failures.push_back("steps_executed mismatch");
    }
    if (!observed.all_finite) {
        out.pass = false;
        out.failures.push_back("observed run is non-finite");
    }

    check_abs("max_abs_u", observed.max_abs_u, baseline.max_abs_u, tolerance.max_abs_u, out);
    check_abs("max_abs_chi", observed.max_abs_chi, baseline.max_abs_chi, tolerance.max_abs_chi, out);
    check_abs("final_u_l2", observed.final_u_l2, baseline.final_u_l2, tolerance.final_u_l2, out);
    check_abs("final_chi_l2", observed.final_chi_l2, baseline.final_chi_l2, tolerance.final_chi_l2, out);
    check_abs("mean_abs_u", observed.mean_abs_u, baseline.mean_abs_u, tolerance.mean_abs_u, out);
    check_abs("mean_abs_chi", observed.mean_abs_chi, baseline.mean_abs_chi, tolerance.mean_abs_chi, out);

    return out;
}

ScenarioRegressionReference default_regression_reference(AberSOEScenario scenario) {
    if (scenario == AberSOEScenario::LinearScalar) {
        ScenarioRegressionReference ref;
        ref.scenario = scenario;
        ref.dt = 0.01;
        ref.steps = 200;
        ref.form = gfe::CouplingForm::NegativeFeedback;
        ref.baseline = RegressionSnapshot{
            1,
            3,
            200,
            true,
            0.983878,
            1.17754,
            0.977558,
            1.67971,
            0.686206,
            0.400748};
        ref.tolerance = RegressionTolerance{0.05, 0.1, 0.1, 0.15, 0.05, 0.05};
        return ref;
    }

    if (scenario == AberSOEScenario::Lorenz63) {
        ScenarioRegressionReference ref;
        ref.scenario = scenario;
        ref.dt = 0.005;
        ref.steps = 200;
        ref.form = gfe::CouplingForm::NegativeFeedback;
        ref.baseline = RegressionSnapshot{
            3,
            3,
            200,
            true,
            50.4428,
            4.19162,
            27.7821,
            0.981049,
            13.8525,
            1.73313};
        ref.tolerance = RegressionTolerance{3.0, 0.7, 2.5, 0.4, 1.0, 0.3};
        return ref;
    }

    if (scenario == AberSOEScenario::Resonant1D) {
        ScenarioRegressionReference ref;
        ref.scenario = scenario;
        ref.dt = 0.01;
        ref.steps = 200;
        ref.form = gfe::CouplingForm::ResonantSecondOrder;
        ref.baseline = RegressionSnapshot{
            2,
            3,
            200,
            true,
            1.0,
            0.903203,
            0.340464,
            1.26805,
            0.528686,
            0.350366};
        ref.tolerance = RegressionTolerance{0.05, 0.1, 0.08, 0.12, 0.05, 0.05};
        return ref;
    }

    throw std::invalid_argument("No default regression reference for scenario");
}

void write_regression_snapshot_csv(const std::string& path, const RegressionSnapshot& snapshot) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open regression snapshot CSV path: " + path);
    }

    out << "state_dim,memory_channels,steps_executed,all_finite,max_abs_u,max_abs_chi,final_u_l2,final_chi_l2,mean_abs_u,mean_abs_chi\n";
    out << snapshot.state_dim << ','
        << snapshot.memory_channels << ','
        << snapshot.steps_executed << ','
        << (snapshot.all_finite ? 1 : 0) << ','
        << std::setprecision(17) << snapshot.max_abs_u << ','
        << snapshot.max_abs_chi << ','
        << snapshot.final_u_l2 << ','
        << snapshot.final_chi_l2 << ','
        << snapshot.mean_abs_u << ','
        << snapshot.mean_abs_chi << '\n';
}

} // namespace abersoe
