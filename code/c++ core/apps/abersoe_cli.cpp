#include "abersoe/abersoe_runtime.hpp"
#include "abersoe/abersoe_regression.hpp"
#include "abersoe/abersoe_scenarios.hpp"
#include "abersoe/abersoe_hooks.hpp"
#include "abersoe/abersoe_learning.hpp"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0
        << " [--scenario linear|lorenz63|resonant1d] [--list-scenarios]"
        << " [--steps N] [--dt X] [--form A|B|C] [--csv PATH] [--diagnostics-csv PATH] [--config-csv PATH]\n"
        << "       [--fit-report-csv PATH] [--fit-backend nnls|prony] [--seed N] [--stochastic-noise-std X]\n"
        << "       [--stochastic-channel I] [--snapshot-csv PATH] [--snapshot-check] [--energy-csv PATH]\n"
        << "       [--hebbian off|oja|plain] [--hebbian-lr X] [--hebbian-decay X] [--hebbian-renorm 0|1] [--no-strict]\n"
        << "\n"
        << "Runs ABERSOE scenarios through gfe::step_augmented wrappers.\n";
}

} // namespace

int main(int argc, char** argv) {
    std::size_t steps = 1000;
    double dt = 0.01;
    gfe::CouplingForm form = gfe::CouplingForm::NegativeFeedback;
    bool strict = true;
    std::string scenario_name = "linear";
    bool list_scenarios_only = false;
    std::string csv_path;
    std::string diagnostics_csv_path;
    std::string config_csv_path;
    std::string fit_report_csv_path;
    std::string snapshot_csv_path;
    std::string energy_csv_path;
    bool snapshot_check = false;
    bool use_prony = false;
    bool hebbian_override = false;
    bool hebbian_enabled = false;
    abersoe::HebbianRule hebbian_rule = abersoe::HebbianRule::OjaNormalized;
    double hebbian_lr = 1e-3;
    double hebbian_decay = 1e-4;
    bool hebbian_renorm = true;
    unsigned long long seed = 123456789ULL;
    double stochastic_noise_std = 0.0;
    std::size_t stochastic_channel = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--steps") {
            if (i + 1 >= argc) {
                std::cerr << "--steps requires a value\n";
                return 2;
            }
            steps = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--scenario") {
            if (i + 1 >= argc) {
                std::cerr << "--scenario requires a value\n";
                return 2;
            }
            scenario_name = argv[++i];
        } else if (arg == "--list-scenarios") {
            list_scenarios_only = true;
        } else if (arg == "--dt") {
            if (i + 1 >= argc) {
                std::cerr << "--dt requires a value\n";
                return 2;
            }
            dt = std::atof(argv[++i]);
        } else if (arg == "--form") {
            if (i + 1 >= argc) {
                std::cerr << "--form requires A, B, or C\n";
                return 2;
            }
            const std::string v = argv[++i];
            if (v == "A" || v == "a") {
                form = gfe::CouplingForm::InputDriven;
            } else if (v == "B" || v == "b") {
                form = gfe::CouplingForm::NegativeFeedback;
            } else if (v == "C" || v == "c") {
                form = gfe::CouplingForm::ResonantSecondOrder;
            } else {
                std::cerr << "--form accepts only A, B, or C\n";
                return 2;
            }
        } else if (arg == "--csv") {
            if (i + 1 >= argc) {
                std::cerr << "--csv requires a path\n";
                return 2;
            }
            csv_path = argv[++i];
        } else if (arg == "--diagnostics-csv") {
            if (i + 1 >= argc) {
                std::cerr << "--diagnostics-csv requires a path\n";
                return 2;
            }
            diagnostics_csv_path = argv[++i];
        } else if (arg == "--config-csv") {
            if (i + 1 >= argc) {
                std::cerr << "--config-csv requires a path\n";
                return 2;
            }
            config_csv_path = argv[++i];
        } else if (arg == "--fit-report-csv") {
            if (i + 1 >= argc) {
                std::cerr << "--fit-report-csv requires a path\n";
                return 2;
            }
            fit_report_csv_path = argv[++i];
        } else if (arg == "--snapshot-csv") {
            if (i + 1 >= argc) {
                std::cerr << "--snapshot-csv requires a path\n";
                return 2;
            }
            snapshot_csv_path = argv[++i];
        } else if (arg == "--snapshot-check") {
            snapshot_check = true;
        } else if (arg == "--energy-csv") {
            if (i + 1 >= argc) {
                std::cerr << "--energy-csv requires a path\n";
                return 2;
            }
            energy_csv_path = argv[++i];
        } else if (arg == "--fit-backend") {
            if (i + 1 >= argc) {
                std::cerr << "--fit-backend requires nnls or prony\n";
                return 2;
            }
            const std::string v = argv[++i];
            if (v == "nnls") {
                use_prony = false;
            } else if (v == "prony") {
                use_prony = true;
            } else {
                std::cerr << "--fit-backend accepts nnls|prony\n";
                return 2;
            }
        } else if (arg == "--hebbian") {
            if (i + 1 >= argc) {
                std::cerr << "--hebbian requires off|oja|plain\n";
                return 2;
            }
            const std::string v = argv[++i];
            hebbian_override = true;
            if (v == "off") {
                hebbian_enabled = false;
            } else if (v == "oja") {
                hebbian_enabled = true;
                hebbian_rule = abersoe::HebbianRule::OjaNormalized;
            } else if (v == "plain") {
                hebbian_enabled = true;
                hebbian_rule = abersoe::HebbianRule::PlainHebbian;
            } else {
                std::cerr << "--hebbian accepts off|oja|plain\n";
                return 2;
            }
        } else if (arg == "--hebbian-lr") {
            if (i + 1 >= argc) {
                std::cerr << "--hebbian-lr requires a value\n";
                return 2;
            }
            hebbian_lr = std::atof(argv[++i]);
        } else if (arg == "--hebbian-decay") {
            if (i + 1 >= argc) {
                std::cerr << "--hebbian-decay requires a value\n";
                return 2;
            }
            hebbian_decay = std::atof(argv[++i]);
        } else if (arg == "--hebbian-renorm") {
            if (i + 1 >= argc) {
                std::cerr << "--hebbian-renorm requires 0 or 1\n";
                return 2;
            }
            const int v = std::atoi(argv[++i]);
            hebbian_renorm = (v != 0);
        } else if (arg == "--seed") {
            if (i + 1 >= argc) {
                std::cerr << "--seed requires a value\n";
                return 2;
            }
            seed = static_cast<unsigned long long>(std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--stochastic-noise-std") {
            if (i + 1 >= argc) {
                std::cerr << "--stochastic-noise-std requires a value\n";
                return 2;
            }
            stochastic_noise_std = std::atof(argv[++i]);
        } else if (arg == "--stochastic-channel") {
            if (i + 1 >= argc) {
                std::cerr << "--stochastic-channel requires a value\n";
                return 2;
            }
            stochastic_channel = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--no-strict") {
            strict = false;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    if (list_scenarios_only) {
        for (const auto& s : abersoe::list_scenarios()) {
            std::cout << s << "\n";
        }
        return 0;
    }

    try {
        const auto scenario = abersoe::parse_scenario(scenario_name);
        abersoe::AberSOEModelSpec model = abersoe::make_scenario_model(scenario, dt, form);
        gfe::AssumptionChecks assumption_checks;
        abersoe::RecordingEnergyObserver energy_observer;
        model.assumption_checks = &assumption_checks;
        model.energy_observer = &energy_observer;
        if (use_prony) {
            model.fitter.backend = gfe::FitBackend::Prony;
            model.fitter.prony.order = 6;
            model.fitter.prony.ridge = 1e-9;
        } else {
            model.fitter.backend = gfe::FitBackend::NnlsProjectedGradient;
        }
        if (hebbian_override) {
            model.learning.enabled = hebbian_enabled;
            model.learning.rule = hebbian_rule;
        }
        model.learning.learning_rate = hebbian_lr;
        model.learning.decay = hebbian_decay;
        model.learning.renormalize = hebbian_renorm;

        abersoe::AberSOEState state0 = abersoe::make_scenario_initial_state(scenario);

        abersoe::AberSOERuntimeConfig cfg;
        cfg.steps = steps;
        cfg.sample_stride = 1;
        cfg.strict_finite = strict;
        cfg.stochastic_forcing = (stochastic_noise_std > 0.0);
        cfg.forcing_noise_std = stochastic_noise_std;
        cfg.stochastic_channel = stochastic_channel;
        cfg.seed = seed;

        const auto result = abersoe::run(model, state0, cfg);

        std::cout << "ABERSOE run summary\n";
        std::cout << "scenario=" << abersoe::scenario_name(scenario) << "\n";
        std::cout << "state_dim=" << result.final_state.u.size() << "\n";
        std::cout << "memory_channels=" << result.final_state.chi.size() << "\n";
        std::cout << "steps_executed=" << result.diagnostics.steps_executed << "\n";
        std::cout << "all_finite=" << (result.diagnostics.all_finite ? 1 : 0) << "\n";
        std::cout << "max_abs_u=" << result.diagnostics.max_abs_u << "\n";
        std::cout << "max_abs_chi=" << result.diagnostics.max_abs_chi << "\n";
        std::cout << "final_u_l2=" << result.diagnostics.final_u_l2 << "\n";
        std::cout << "final_chi_l2=" << result.diagnostics.final_chi_l2 << "\n";
        std::cout << "mean_abs_u=" << result.diagnostics.mean_abs_u << "\n";
        std::cout << "mean_abs_chi=" << result.diagnostics.mean_abs_chi << "\n";
        std::cout << "seed_used=" << result.seed_used << "\n";
        std::cout << "stochastic_forcing_applied=" << (result.stochastic_forcing_applied ? 1 : 0) << "\n";
        std::cout << "hooks_assumption_attached=" << (result.hooks.assumption_checks_attached ? 1 : 0) << "\n";
        std::cout << "hooks_energy_attached=" << (result.hooks.energy_observer_attached ? 1 : 0) << "\n";
        std::cout << "hooks_energy_events=" << result.hooks.energy_events << "\n";
        std::cout << "hooks_assumption_note=" << result.hooks.assumption_note << "\n";
        std::cout << "hebbian_enabled=" << (model.learning.enabled ? 1 : 0) << "\n";
        std::cout << "hebbian_rule=" << abersoe::hebbian_rule_name(model.learning.rule) << "\n";
        std::cout << "hebbian_learning_updates=" << result.learning_updates << "\n";
        if (!result.final_kernel_w.empty()) {
            std::cout << "hebbian_final_w0=" << result.final_kernel_w[0] << "\n";
        }

        if (!csv_path.empty()) {
            abersoe::write_run_csv(csv_path, result);
            std::cout << "csv=" << csv_path << "\n";
        }
        if (!energy_csv_path.empty()) {
            abersoe::write_energy_samples_csv(energy_csv_path, energy_observer.samples());
            std::cout << "energy_csv=" << energy_csv_path << "\n";
        }
        if (!diagnostics_csv_path.empty()) {
            abersoe::write_diagnostics_csv(diagnostics_csv_path, result.diagnostics);
            std::cout << "diagnostics_csv=" << diagnostics_csv_path << "\n";
        }
        if (!config_csv_path.empty()) {
            abersoe::write_config_csv(config_csv_path, result.config);
            std::cout << "config_csv=" << config_csv_path << "\n";
        }
        if (!fit_report_csv_path.empty()) {
            // Backend-agnostic fit report on a synthetic reference generated from model kernel.
            gfe::Vec t_ref;
            gfe::Vec y_ref;
            t_ref.reserve(201);
            y_ref.reserve(201);
            for (int i = 0; i <= 200; ++i) {
                const double t = 0.05 * static_cast<double>(i);
                t_ref.push_back(t);
                double val = 0.0;
                for (std::size_t j = 0; j < model.kernel.gamma.size(); ++j) {
                    val += model.kernel.w[j] * std::exp(-model.kernel.gamma[j] * t);
                }
                y_ref.push_back(val);
            }
            const auto fit_report = abersoe::build_fit_report(t_ref, y_ref, model);
            abersoe::write_fit_report_csv(fit_report_csv_path, fit_report);
            std::cout << "fit_report_csv=" << fit_report_csv_path << "\n";
            std::cout << "fit_backend=" << fit_report.backend << "\n";
            std::cout << "fit_ok=" << (fit_report.ok ? 1 : 0) << "\n";
            std::cout << "fit_l1_error=" << fit_report.l1_error << "\n";
            std::cout << "fit_modes_kept=" << fit_report.modes_kept << "\n";
        }
        const auto snapshot = abersoe::make_regression_snapshot(result);
        if (!snapshot_csv_path.empty()) {
            abersoe::write_regression_snapshot_csv(snapshot_csv_path, snapshot);
            std::cout << "snapshot_csv=" << snapshot_csv_path << "\n";
        }
        if (snapshot_check) {
            const auto ref = abersoe::default_regression_reference(scenario);
            const bool cfg_match = (steps == ref.steps) &&
                                   (std::fabs(dt - ref.dt) <= 1e-12) &&
                                   (form == ref.form);
            if (!cfg_match) {
                std::cout << "snapshot_check_pass=0\n";
                std::cout << "snapshot_check_failure=config mismatch for reference baseline\n";
                std::cout << "snapshot_check_expected_steps=" << ref.steps << "\n";
                std::cout << "snapshot_check_expected_dt=" << ref.dt << "\n";
                const char expected_form =
                    (ref.form == gfe::CouplingForm::InputDriven)
                        ? 'A'
                        : ((ref.form == gfe::CouplingForm::NegativeFeedback) ? 'B' : 'C');
                std::cout << "snapshot_check_expected_form=" << expected_form << "\n";
                if (strict) {
                    return 1;
                }
            }
            if (cfg_match) {
                const auto chk = abersoe::check_regression_snapshot(snapshot, ref.baseline, ref.tolerance);
                std::cout << "snapshot_check_pass=" << (chk.pass ? 1 : 0) << "\n";
                for (const auto& f : chk.failures) {
                    std::cout << "snapshot_check_failure=" << f << "\n";
                }
                if (strict && !chk.pass) {
                    return 1;
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "abersoe_cli failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
