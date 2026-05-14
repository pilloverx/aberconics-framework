#include "gfe/gfe_c_api.h"
#include "gfe/gfe_core.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool almost_equal(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol * (1.0 + std::max(std::fabs(a), std::fabs(b)));
}

} // namespace

int main() {
    try {
        {
            const auto spectral_opts = gfe_c_default_spectral_unit_options();
            if (spectral_opts.deff_policy != GFE_C_DEFF_PAPER_EFFECTIVE_DIMENSION ||
                spectral_opts.mcap_policy != GFE_C_MCAP_PAPER_MEAN_DEPTH) {
                std::cerr << "unexpected default spectral-unit policy values\n";
                return 1;
            }

            const auto abersoe_cfg = gfe_c_abersoe_default_runtime_config();
            if (abersoe_cfg.steps == 0 || abersoe_cfg.sample_stride == 0 || !abersoe_cfg.strict_finite) {
                std::cerr << "unexpected default abersoe runtime config\n";
                return 1;
            }

            const auto hier_cfg = gfe_c_hierarchical_default_runtime_config();
            if (hier_cfg.steps == 0 || hier_cfg.sample_stride == 0 || !hier_cfg.strict_finite) {
                std::cerr << "unexpected default hierarchical runtime config\n";
                return 1;
            }
        }

        std::vector<double> t;
        for (int i = 0; i <= 80; ++i) {
            t.push_back(0.1 * static_cast<double>(i));
        }

        const gfe::Vec gamma_true{0.8, 0.12};
        const gfe::Vec w_true{0.65, 0.35};
        const auto A = gfe::design_matrix(gamma_true, t);

        std::vector<double> y(t.size(), 0.0);
        for (std::size_t i = 0; i < y.size(); ++i) {
            y[i] = A[i][0] * w_true[0] + A[i][1] * w_true[1];
        }

        gfe_c_soe_fit_options c_opts = gfe_c_default_soe_fit_options();
        c_opts.backend = GFE_C_FIT_BACKEND_NNLS_PG;
        c_opts.n_basis = 14;
        c_opts.gamma_min = 1e-2;
        c_opts.gamma_max = 10.0;
        c_opts.threshold = 0.0;
        c_opts.lr = 1e-3;
        c_opts.iters = 5000;

        gfe_c_backend_config c_backend = gfe_c_default_backend_config();
        c_backend.preferred = GFE_C_COMPUTE_BACKEND_GPU_CUDA;
        c_backend.allow_fallback = 1;
        c_backend.fallback = GFE_C_COMPUTE_BACKEND_CPU_REF;

        std::vector<double> gamma_out(64, 0.0);
        std::vector<double> w_out(64, 0.0);
        std::vector<double> fit_out(t.size(), 0.0);
        std::size_t gamma_size = 0;
        std::size_t w_size = 0;
        std::size_t fit_size = 0;
        char errbuf[256] = {0};

        gfe_c_fit_result_view out{};
        out.gamma_fit = gamma_out.data();
        out.gamma_fit_capacity = gamma_out.size();
        out.gamma_fit_size = &gamma_size;
        out.w_fit = w_out.data();
        out.w_fit_capacity = w_out.size();
        out.w_fit_size = &w_size;
        out.fit = fit_out.data();
        out.fit_capacity = fit_out.size();
        out.fit_size = &fit_size;

        const int rc = gfe_c_fit_soe_kernel(t.data(),
                                            t.size(),
                                            y.data(),
                                            y.size(),
                                            &c_opts,
                                            &c_backend,
                                            &out,
                                            errbuf,
                                            sizeof(errbuf));
        if (rc != GFE_C_STATUS_OK) {
            std::cerr << "gfe_c_fit_soe_kernel failed: " << errbuf << "\n";
            return 1;
        }

        gfe::SoeFitOptions cpp_opts;
        cpp_opts.backend = gfe::FitBackend::NnlsProjectedGradient;
        cpp_opts.n_basis = c_opts.n_basis;
        cpp_opts.gamma_min = c_opts.gamma_min;
        cpp_opts.gamma_max = c_opts.gamma_max;
        cpp_opts.threshold = c_opts.threshold;
        cpp_opts.lr = c_opts.lr;
        cpp_opts.iters = c_opts.iters;

        gfe::BackendConfig cpp_backend;
        cpp_backend.preferred = gfe::ComputeBackend::GpuCuda;
        cpp_backend.allow_fallback = true;
        cpp_backend.fallback = gfe::ComputeBackend::CpuReference;

        const auto cpp_fit = gfe::fit_soe_kernel(t, y, cpp_opts, cpp_backend);
        if (gamma_size != cpp_fit.gamma_fit.size() || w_size != cpp_fit.w_fit.size() || fit_size != cpp_fit.fit.size()) {
            std::cerr << "C ABI/C++ result sizes mismatch\n";
            return 1;
        }
        for (std::size_t i = 0; i < fit_size; ++i) {
            if (!almost_equal(fit_out[i], cpp_fit.fit[i], 1e-9)) {
                std::cerr << "C ABI/C++ fit mismatch\n";
                return 1;
            }
        }

        // Buffer-too-small contract check.
        std::vector<double> tiny_fit(4, 0.0);
        std::size_t tiny_fit_size = 0;
        out.fit = tiny_fit.data();
        out.fit_capacity = tiny_fit.size();
        out.fit_size = &tiny_fit_size;
        const int rc_small = gfe_c_fit_soe_kernel(t.data(),
                                                  t.size(),
                                                  y.data(),
                                                  y.size(),
                                                  &c_opts,
                                                  &c_backend,
                                                  &out,
                                                  errbuf,
                                                  sizeof(errbuf));
        if (rc_small != GFE_C_STATUS_BUFFER_TOO_SMALL) {
            std::cerr << "Expected BUFFER_TOO_SMALL\n";
            return 1;
        }
        if (tiny_fit_size != cpp_fit.fit.size()) {
            std::cerr << "Expected fit_size to report required size\n";
            return 1;
        }

        {
            const gfe_c_memory_kernel_view kernel_view{
                gamma_true.data(),
                gamma_true.size(),
                w_true.data(),
                w_true.size(),
            };
            char errbuf2[256] = {0};
            const int valid_rc = gfe_c_validate_memory_kernel(&kernel_view, errbuf2, sizeof(errbuf2));
            if (valid_rc != GFE_C_STATUS_OK) {
                std::cerr << "gfe_c_validate_memory_kernel failed: " << errbuf2 << "\n";
                return 1;
            }

            gfe_c_spectral_units spectral{};
            const auto spectral_opts = gfe_c_default_spectral_unit_options();
            const int spectral_rc = gfe_c_spectral_units_from_kernel(
                &kernel_view, &spectral_opts, &spectral, errbuf2, sizeof(errbuf2));
            if (spectral_rc != GFE_C_STATUS_OK) {
                std::cerr << "gfe_c_spectral_units_from_kernel failed: " << errbuf2 << "\n";
                return 1;
            }
            const auto cpp_spectral = gfe::spectral_units(w_true, gamma_true);
            if (!almost_equal(spectral.mcap, cpp_spectral.Mcap) ||
                !almost_equal(spectral.mscale, cpp_spectral.Mscale) ||
                !almost_equal(spectral.deff, cpp_spectral.Deff)) {
                std::cerr << "C ABI/C++ spectral-units mismatch\n";
                return 1;
            }
        }

        {
            char name_buf[64] = {0};
            std::size_t name_size = 0;
            gfe_c_string_result_view out_name{name_buf, sizeof(name_buf), &name_size};
            char errbuf3[256] = {0};
            const int name_rc =
                gfe_c_fit_backend_name(GFE_C_FIT_BACKEND_NNLS_PG, &out_name, errbuf3, sizeof(errbuf3));
            if (name_rc != GFE_C_STATUS_OK || std::string(name_buf) != "nnls_pg") {
                std::cerr << "gfe_c_fit_backend_name failed\n";
                return 1;
            }

            char short_header_buf[32] = {0};
            std::size_t short_header_size = 0;
            gfe_c_string_result_view short_header{short_header_buf, sizeof(short_header_buf), &short_header_size};
            const int short_header_rc =
                gfe_c_hierarchy_renorm_csv_header(&short_header, errbuf3, sizeof(errbuf3));
            if (short_header_rc != GFE_C_STATUS_BUFFER_TOO_SMALL || short_header_size == 0) {
                std::cerr << "gfe_c_hierarchy_renorm_csv_header should report buffer-too-small with required size\n";
                return 1;
            }

            std::vector<char> header_buf(short_header_size + 1, '\0');
            std::size_t header_size = 0;
            gfe_c_string_result_view out_header{header_buf.data(), header_buf.size(), &header_size};
            const int header_rc =
                gfe_c_hierarchy_renorm_csv_header(&out_header, errbuf3, sizeof(errbuf3));
            if (header_rc != GFE_C_STATUS_OK) {
                std::cerr << "gfe_c_hierarchy_renorm_csv_header failed: " << errbuf3 << "\n";
                return 1;
            }
            if (std::string(header_buf.data()).find("input_level,bath_level,direct_level") != 0) {
                std::cerr << "unexpected renorm CSV header content\n";
                return 1;
            }
        }

        {
            if (gfe_c_abersoe_scenario_count() < 3) {
                std::cerr << "unexpected abersoe scenario count\n";
                return 1;
            }

            char scenario_name_buf[64] = {0};
            std::size_t scenario_name_size = 0;
            gfe_c_string_result_view out_scenario_name{
                scenario_name_buf, sizeof(scenario_name_buf), &scenario_name_size};
            char errbuf4[256] = {0};
            const int scenario_name_rc = gfe_c_abersoe_scenario_name(
                GFE_C_ABERSOE_SCENARIO_LORENZ63, &out_scenario_name, errbuf4, sizeof(errbuf4));
            if (scenario_name_rc != GFE_C_STATUS_OK || std::string(scenario_name_buf) != "lorenz63") {
                std::cerr << "gfe_c_abersoe_scenario_name failed\n";
                return 1;
            }

            auto run_cfg = gfe_c_abersoe_default_runtime_config();
            run_cfg.steps = 20;
            run_cfg.sample_stride = 5;
            run_cfg.strict_finite = 1;

            std::vector<double> final_u(8, 0.0);
            std::vector<double> final_chi(8, 0.0);
            double final_t = 0.0;
            std::size_t final_u_size = 0;
            std::size_t final_chi_size = 0;
            gfe_c_state_mut_view final_state{
                final_u.data(),
                final_u.size(),
                &final_u_size,
                final_chi.data(),
                final_chi.size(),
                &final_chi_size,
                &final_t,
            };

            std::vector<double> sample_t(16, 0.0);
            std::vector<double> sample_u0(16, 0.0);
            std::vector<double> sample_chi0(16, 0.0);
            std::vector<double> sample_u_l2(16, 0.0);
            std::vector<double> sample_chi_l2(16, 0.0);
            std::size_t sample_t_size = 0;
            std::size_t sample_u0_size = 0;
            std::size_t sample_chi0_size = 0;
            std::size_t sample_u_l2_size = 0;
            std::size_t sample_chi_l2_size = 0;
            gfe_c_abersoe_sample_summary_view samples{
                sample_t.data(),
                sample_t.size(),
                &sample_t_size,
                sample_u0.data(),
                sample_u0.size(),
                &sample_u0_size,
                sample_chi0.data(),
                sample_chi0.size(),
                &sample_chi0_size,
                sample_u_l2.data(),
                sample_u_l2.size(),
                &sample_u_l2_size,
                sample_chi_l2.data(),
                sample_chi_l2.size(),
                &sample_chi_l2_size,
            };

            gfe_c_abersoe_diagnostics diag{};
            gfe_c_abersoe_config_record cfg_record{};
            const int run_rc = gfe_c_abersoe_run_scenario(
                GFE_C_ABERSOE_SCENARIO_LINEAR,
                0.01,
                GFE_C_COUPLING_FORM_B,
                &run_cfg,
                &final_state,
                &diag,
                &cfg_record,
                &samples,
                errbuf4,
                sizeof(errbuf4));
            if (run_rc != GFE_C_STATUS_OK) {
                std::cerr << "gfe_c_abersoe_run_scenario failed: " << errbuf4 << "\n";
                return 1;
            }
            if (final_u_size != 1 || final_chi_size != 3) {
                std::cerr << "unexpected final-state shape from abersoe scenario run\n";
                return 1;
            }
            if (diag.steps_executed != run_cfg.steps || !diag.all_finite) {
                std::cerr << "unexpected diagnostics from abersoe scenario run\n";
                return 1;
            }
            if (cfg_record.steps != run_cfg.steps || cfg_record.sample_stride != run_cfg.sample_stride) {
                std::cerr << "unexpected config record from abersoe scenario run\n";
                return 1;
            }
            if (sample_t_size == 0 || sample_t_size != sample_u0_size ||
                sample_t_size != sample_chi0_size || sample_t_size != sample_u_l2_size ||
                sample_t_size != sample_chi_l2_size) {
                std::cerr << "unexpected sample-summary shape from abersoe scenario run\n";
                return 1;
            }
            if (!(final_t > 0.0) || !(sample_t[sample_t_size - 1] > 0.0)) {
                std::cerr << "expected positive final/sample times from abersoe scenario run\n";
                return 1;
            }

            std::vector<double> custom_gamma{1.4, 0.6, 0.2};
            std::vector<double> custom_w{0.45, 0.35, 0.20};
            gfe_c_memory_kernel_view kernel_override{
                custom_gamma.data(),
                custom_gamma.size(),
                custom_w.data(),
                custom_w.size(),
            };

            std::vector<double> init_u{2.0, -1.0, 8.0};
            std::vector<double> init_chi{0.1, -0.2, 0.05};
            gfe_c_state_view initial_state_override{
                init_u.data(),
                init_u.size(),
                init_chi.data(),
                init_chi.size(),
                0.25,
            };

            std::vector<double> active_gamma(8, 0.0);
            std::vector<double> active_w(8, 0.0);
            std::size_t active_gamma_size = 0;
            std::size_t active_w_size = 0;
            gfe_c_memory_kernel_mut_view active_kernel{
                active_gamma.data(),
                active_gamma.size(),
                &active_gamma_size,
                active_w.data(),
                active_w.size(),
                &active_w_size,
            };

            std::vector<double> traj_t(16, 0.0);
            std::vector<double> traj_u(16 * 3, 0.0);
            std::vector<double> traj_chi(16 * 3, 0.0);
            std::size_t traj_t_size = 0;
            std::size_t traj_u_size = 0;
            std::size_t traj_u_dim = 0;
            std::size_t traj_chi_size = 0;
            std::size_t traj_chi_dim = 0;
            gfe_c_abersoe_trajectory_view trajectory{
                traj_t.data(),
                traj_t.size(),
                &traj_t_size,
                traj_u.data(),
                traj_u.size(),
                &traj_u_size,
                &traj_u_dim,
                traj_chi.data(),
                traj_chi.size(),
                &traj_chi_size,
                &traj_chi_dim,
            };

            const int custom_run_rc = gfe_c_abersoe_run_scenario_with_overrides(
                GFE_C_ABERSOE_SCENARIO_LORENZ63,
                0.01,
                GFE_C_COUPLING_FORM_B,
                &run_cfg,
                &initial_state_override,
                &kernel_override,
                &final_state,
                &active_kernel,
                &diag,
                &cfg_record,
                &samples,
                &trajectory,
                errbuf4,
                sizeof(errbuf4));
            if (custom_run_rc != GFE_C_STATUS_OK) {
                std::cerr << "gfe_c_abersoe_run_scenario_with_overrides failed: " << errbuf4 << "\n";
                return 1;
            }
            if (final_u_size != 3 || final_chi_size != custom_gamma.size()) {
                std::cerr << "unexpected final-state shape from custom Lorenz run\n";
                return 1;
            }
            if (active_gamma_size != custom_gamma.size() || active_w_size != custom_w.size()) {
                std::cerr << "unexpected active-kernel shape from custom Lorenz run\n";
                return 1;
            }
            for (std::size_t i = 0; i < custom_gamma.size(); ++i) {
                if (std::abs(active_gamma[i] - custom_gamma[i]) > 1e-12 ||
                    std::abs(active_w[i] - custom_w[i]) > 1e-12) {
                    std::cerr << "active kernel did not round-trip custom override\n";
                    return 1;
                }
            }
            if (traj_t_size != sample_t_size || traj_u_dim != 3 || traj_chi_dim != custom_gamma.size()) {
                std::cerr << "unexpected trajectory dimensions from custom Lorenz run\n";
                return 1;
            }
            if (traj_u_size != traj_t_size * traj_u_dim || traj_chi_size != traj_t_size * traj_chi_dim) {
                std::cerr << "unexpected flattened trajectory sizes from custom Lorenz run\n";
                return 1;
            }
            if (std::abs(traj_t.front() - initial_state_override.t) > 1e-12) {
                std::cerr << "trajectory should preserve custom initial time\n";
                return 1;
            }
        }

        {
            if (gfe_c_hierarchical_scenario_count() < 2) {
                std::cerr << "unexpected hierarchical scenario count\n";
                return 1;
            }

            char scenario_name_buf[64] = {0};
            std::size_t scenario_name_size = 0;
            gfe_c_string_result_view out_scenario_name{
                scenario_name_buf, sizeof(scenario_name_buf), &scenario_name_size};
            char errbuf5[256] = {0};
            const int scenario_name_rc = gfe_c_hierarchical_scenario_name(
                GFE_C_HIERARCHICAL_SCENARIO_THREE_LEVEL_HOMEOSTATIC,
                &out_scenario_name,
                errbuf5,
                sizeof(errbuf5));
            if (scenario_name_rc != GFE_C_STATUS_OK || std::string(scenario_name_buf) != "three_level_homeostatic") {
                std::cerr << "gfe_c_hierarchical_scenario_name failed\n";
                return 1;
            }

            auto hier_cfg = gfe_c_hierarchical_default_runtime_config();
            hier_cfg.steps = 24;
            hier_cfg.sample_stride = 6;
            hier_cfg.strict_finite = 1;

            std::vector<double> t_buf(4, 0.0);
            std::vector<std::size_t> state_dim_buf(4, 0);
            std::vector<std::size_t> memory_channels_buf(4, 0);
            std::vector<double> u0_buf(4, 0.0);
            std::vector<double> chi0_buf(4, 0.0);
            std::vector<double> w0_buf(4, 0.0);
            std::vector<double> gamma0_buf(4, 0.0);
            std::vector<double> mcap_buf(4, 0.0);
            std::vector<double> mscale_buf(4, 0.0);
            std::vector<double> hmem_buf(4, 0.0);
            std::vector<double> deff_buf(4, 0.0);
            std::size_t t_size = 0;
            std::size_t state_dim_size = 0;
            std::size_t memory_channels_size = 0;
            std::size_t u0_size = 0;
            std::size_t chi0_size = 0;
            std::size_t w0_size = 0;
            std::size_t gamma0_size = 0;
            std::size_t mcap_size = 0;
            std::size_t mscale_size = 0;
            std::size_t hmem_size = 0;
            std::size_t deff_size = 0;
            gfe_c_hierarchical_level_summary_view levels{
                t_buf.data(), t_buf.size(), &t_size,
                state_dim_buf.data(), state_dim_buf.size(), &state_dim_size,
                memory_channels_buf.data(), memory_channels_buf.size(), &memory_channels_size,
                u0_buf.data(), u0_buf.size(), &u0_size,
                chi0_buf.data(), chi0_buf.size(), &chi0_size,
                w0_buf.data(), w0_buf.size(), &w0_size,
                gamma0_buf.data(), gamma0_buf.size(), &gamma0_size,
                mcap_buf.data(), mcap_buf.size(), &mcap_size,
                mscale_buf.data(), mscale_buf.size(), &mscale_size,
                hmem_buf.data(), hmem_buf.size(), &hmem_size,
                deff_buf.data(), deff_buf.size(), &deff_size,
            };
            gfe_c_hierarchical_run_diagnostics hier_diag{};
            const int hier_rc = gfe_c_hierarchical_run_scenario(
                GFE_C_HIERARCHICAL_SCENARIO_THREE_LEVEL_HOMEOSTATIC,
                1,
                1,
                &hier_cfg,
                &levels,
                &hier_diag,
                errbuf5,
                sizeof(errbuf5));
            if (hier_rc != GFE_C_STATUS_OK) {
                std::cerr << "gfe_c_hierarchical_run_scenario failed: " << errbuf5 << "\n";
                return 1;
            }
            if (hier_diag.steps_executed != hier_cfg.steps || !hier_diag.all_finite || hier_diag.level_count != 3) {
                std::cerr << "unexpected hierarchical diagnostics\n";
                return 1;
            }
            if (t_size != 3 || u0_size != 3 || deff_size != 3 || state_dim_size != 3 || memory_channels_size != 3) {
                std::cerr << "unexpected hierarchical summary shape\n";
                return 1;
            }
            if (!(t_buf[2] > 0.0) || !(deff_buf[0] > 0.0)) {
                std::cerr << "expected positive hierarchy summary values\n";
                return 1;
            }

            std::vector<std::size_t> shift_level_idx(4, 0);
            std::vector<double> shift_delta_mcap(4, 0.0);
            std::vector<double> shift_delta_mscale(4, 0.0);
            std::vector<double> shift_delta_hmem(4, 0.0);
            std::vector<double> shift_delta_deff(4, 0.0);
            std::vector<int> shift_changed(4, 0);
            std::size_t shift_level_idx_size = 0;
            std::size_t shift_delta_mcap_size = 0;
            std::size_t shift_delta_mscale_size = 0;
            std::size_t shift_delta_hmem_size = 0;
            std::size_t shift_delta_deff_size = 0;
            std::size_t shift_changed_size = 0;
            gfe_c_hierarchical_kernel_shift_view kernel_shifts{
                shift_level_idx.data(), shift_level_idx.size(), &shift_level_idx_size,
                shift_delta_mcap.data(), shift_delta_mcap.size(), &shift_delta_mcap_size,
                shift_delta_mscale.data(), shift_delta_mscale.size(), &shift_delta_mscale_size,
                shift_delta_hmem.data(), shift_delta_hmem.size(), &shift_delta_hmem_size,
                shift_delta_deff.data(), shift_delta_deff.size(), &shift_delta_deff_size,
                shift_changed.data(), shift_changed.size(), &shift_changed_size,
            };

            std::vector<int> relation_type(8, 0);
            std::vector<std::size_t> relation_source(8, 0);
            std::vector<std::size_t> relation_target(8, 0);
            std::vector<double> relation_delta_mcap(8, 0.0);
            std::vector<double> relation_delta_mscale(8, 0.0);
            std::vector<double> relation_delta_hmem(8, 0.0);
            std::vector<double> relation_delta_deff(8, 0.0);
            std::vector<int> relation_target_mcap(8, 0);
            std::vector<int> relation_target_mscale(8, 0);
            std::size_t relation_type_size = 0;
            std::size_t relation_source_size = 0;
            std::size_t relation_target_size = 0;
            std::size_t relation_delta_mcap_size = 0;
            std::size_t relation_delta_mscale_size = 0;
            std::size_t relation_delta_hmem_size = 0;
            std::size_t relation_delta_deff_size = 0;
            std::size_t relation_target_mcap_size = 0;
            std::size_t relation_target_mscale_size = 0;
            gfe_c_hierarchical_relation_view relations{
                relation_type.data(), relation_type.size(), &relation_type_size,
                relation_source.data(), relation_source.size(), &relation_source_size,
                relation_target.data(), relation_target.size(), &relation_target_size,
                relation_delta_mcap.data(), relation_delta_mcap.size(), &relation_delta_mcap_size,
                relation_delta_mscale.data(), relation_delta_mscale.size(), &relation_delta_mscale_size,
                relation_delta_hmem.data(), relation_delta_hmem.size(), &relation_delta_hmem_size,
                relation_delta_deff.data(), relation_delta_deff.size(), &relation_delta_deff_size,
                relation_target_mcap.data(), relation_target_mcap.size(), &relation_target_mcap_size,
                relation_target_mscale.data(), relation_target_mscale.size(), &relation_target_mscale_size,
            };

            std::vector<int> warning_severity(8, 0);
            std::vector<int> warning_code(8, 0);
            std::vector<int> warning_metric(8, 0);
            std::vector<double> warning_observed(8, 0.0);
            std::vector<double> warning_threshold(8, 0.0);
            std::vector<std::size_t> warning_source(8, 0);
            std::vector<std::size_t> warning_target(8, 0);
            std::vector<int> warning_actionable(8, 0);
            std::size_t warning_severity_size = 0;
            std::size_t warning_code_size = 0;
            std::size_t warning_metric_size = 0;
            std::size_t warning_observed_size = 0;
            std::size_t warning_threshold_size = 0;
            std::size_t warning_source_size = 0;
            std::size_t warning_target_size = 0;
            std::size_t warning_actionable_size = 0;
            gfe_c_hierarchical_warning_view warnings{
                warning_severity.data(), warning_severity.size(), &warning_severity_size,
                warning_code.data(), warning_code.size(), &warning_code_size,
                warning_metric.data(), warning_metric.size(), &warning_metric_size,
                warning_observed.data(), warning_observed.size(), &warning_observed_size,
                warning_threshold.data(), warning_threshold.size(), &warning_threshold_size,
                warning_source.data(), warning_source.size(), &warning_source_size,
                warning_target.data(), warning_target.size(), &warning_target_size,
                warning_actionable.data(), warning_actionable.size(), &warning_actionable_size,
            };
            gfe_c_hierarchical_cross_level_contract contract{};
            const int cross_rc = gfe_c_hierarchical_get_cross_level_report(
                GFE_C_HIERARCHICAL_SCENARIO_THREE_LEVEL_HOMEOSTATIC,
                1,
                1,
                &hier_cfg,
                &kernel_shifts,
                &relations,
                &warnings,
                &contract,
                errbuf5,
                sizeof(errbuf5));
            if (cross_rc != GFE_C_STATUS_OK) {
                std::cerr << "gfe_c_hierarchical_get_cross_level_report failed: " << errbuf5 << "\n";
                return 1;
            }
            if (!contract.all_finite_pass || shift_level_idx_size != 3 || relation_type_size < 2) {
                std::cerr << "unexpected cross-level report shape or contract\n";
                return 1;
            }

            std::vector<std::size_t> renorm_input(4, 0);
            std::vector<std::size_t> renorm_bath(4, 0);
            std::vector<std::size_t> renorm_direct(4, 0);
            std::vector<int> renorm_diag(4, 0);
            std::vector<std::size_t> renorm_raw_channels(4, 0);
            std::vector<std::size_t> renorm_output_channels(4, 0);
            std::vector<std::size_t> renorm_pruned_channels(4, 0);
            std::vector<double> renorm_retained(4, 0.0);
            std::vector<double> renorm_fit_l2(4, 0.0);
            std::vector<double> renorm_pruned_fit_l2(4, 0.0);
            std::vector<double> renorm_fit_linf(4, 0.0);
            std::vector<double> renorm_pruned_fit_linf(4, 0.0);
            std::vector<double> renorm_delta_mcap(4, 0.0);
            std::vector<double> renorm_delta_mscale(4, 0.0);
            std::vector<double> renorm_delta_deff(4, 0.0);
            std::vector<int> renorm_consistency(4, 0);
            std::vector<int> renorm_mcap_pass(4, 0);
            std::vector<int> renorm_mscale_pass(4, 0);
            std::vector<int> renorm_deff_pass(4, 0);
            std::size_t renorm_input_size = 0;
            std::size_t renorm_bath_size = 0;
            std::size_t renorm_direct_size = 0;
            std::size_t renorm_diag_size = 0;
            std::size_t renorm_raw_channels_size = 0;
            std::size_t renorm_output_channels_size = 0;
            std::size_t renorm_pruned_channels_size = 0;
            std::size_t renorm_retained_size = 0;
            std::size_t renorm_fit_l2_size = 0;
            std::size_t renorm_pruned_fit_l2_size = 0;
            std::size_t renorm_fit_linf_size = 0;
            std::size_t renorm_pruned_fit_linf_size = 0;
            std::size_t renorm_delta_mcap_size = 0;
            std::size_t renorm_delta_mscale_size = 0;
            std::size_t renorm_delta_deff_size = 0;
            std::size_t renorm_consistency_size = 0;
            std::size_t renorm_mcap_pass_size = 0;
            std::size_t renorm_mscale_pass_size = 0;
            std::size_t renorm_deff_pass_size = 0;
            gfe_c_hierarchical_renorm_analysis_view renorm{
                renorm_input.data(), renorm_input.size(), &renorm_input_size,
                renorm_bath.data(), renorm_bath.size(), &renorm_bath_size,
                renorm_direct.data(), renorm_direct.size(), &renorm_direct_size,
                renorm_diag.data(), renorm_diag.size(), &renorm_diag_size,
                renorm_raw_channels.data(), renorm_raw_channels.size(), &renorm_raw_channels_size,
                renorm_output_channels.data(), renorm_output_channels.size(), &renorm_output_channels_size,
                renorm_pruned_channels.data(), renorm_pruned_channels.size(), &renorm_pruned_channels_size,
                renorm_retained.data(), renorm_retained.size(), &renorm_retained_size,
                renorm_fit_l2.data(), renorm_fit_l2.size(), &renorm_fit_l2_size,
                renorm_pruned_fit_l2.data(), renorm_pruned_fit_l2.size(), &renorm_pruned_fit_l2_size,
                renorm_fit_linf.data(), renorm_fit_linf.size(), &renorm_fit_linf_size,
                renorm_pruned_fit_linf.data(), renorm_pruned_fit_linf.size(), &renorm_pruned_fit_linf_size,
                renorm_delta_mcap.data(), renorm_delta_mcap.size(), &renorm_delta_mcap_size,
                renorm_delta_mscale.data(), renorm_delta_mscale.size(), &renorm_delta_mscale_size,
                renorm_delta_deff.data(), renorm_delta_deff.size(), &renorm_delta_deff_size,
                renorm_consistency.data(), renorm_consistency.size(), &renorm_consistency_size,
                renorm_mcap_pass.data(), renorm_mcap_pass.size(), &renorm_mcap_pass_size,
                renorm_mscale_pass.data(), renorm_mscale_pass.size(), &renorm_mscale_pass_size,
                renorm_deff_pass.data(), renorm_deff_pass.size(), &renorm_deff_pass_size,
            };
            const int renorm_rc = gfe_c_hierarchical_get_renorm_report(
                GFE_C_HIERARCHICAL_SCENARIO_THREE_LEVEL_HOMEOSTATIC,
                1,
                1,
                &renorm,
                errbuf5,
                sizeof(errbuf5));
            if (renorm_rc != GFE_C_STATUS_OK) {
                std::cerr << "gfe_c_hierarchical_get_renorm_report failed: " << errbuf5 << "\n";
                return 1;
            }
            if (renorm_input_size != 2 || renorm_retained[0] < 0.0 || renorm_retained[0] > 1.0) {
                std::cerr << "unexpected renorm report shape or retained fraction\n";
                return 1;
            }
        }

        {
            std::vector<double> fast_gamma{1.0, 0.4};
            std::vector<double> fast_w{0.65, 0.35};
            std::vector<double> fast_u{1.0};
            std::vector<double> fast_chi{0.0, 0.0};
            std::vector<double> fast_decay{0.25};
            std::vector<double> fast_force{0.0};
            gfe_c_hierarchical_chain_level_spec level0{
                "custom_fast",
                {fast_gamma.data(), fast_gamma.size(), fast_w.data(), fast_w.size()},
                {fast_u.data(), fast_u.size(), fast_chi.data(), fast_chi.size(), 0.0},
                0.01,
                fast_decay.data(),
                fast_decay.size(),
                fast_force.data(),
                fast_force.size(),
                GFE_C_COUPLING_FORM_B,
                0,
            };

            std::vector<double> slow_gamma{0.7, 0.15};
            std::vector<double> slow_w{0.6, 0.4};
            std::vector<double> slow_u{0.1};
            std::vector<double> slow_chi{0.0, 0.0};
            std::vector<double> slow_decay{0.08};
            std::vector<double> slow_force{0.01};
            gfe_c_hierarchical_chain_level_spec level1{
                "custom_slow",
                {slow_gamma.data(), slow_gamma.size(), slow_w.data(), slow_w.size()},
                {slow_u.data(), slow_u.size(), slow_chi.data(), slow_chi.size(), 0.0},
                0.01,
                slow_decay.data(),
                slow_decay.size(),
                slow_force.data(),
                slow_force.size(),
                GFE_C_COUPLING_FORM_B,
                0,
            };

            std::vector<gfe_c_hierarchical_chain_level_spec> chain_levels{level0, level1};
            std::vector<gfe_c_hierarchical_chain_edge_spec> chain_edges{
                {0, 1, GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP, 0.35, 1},
                {1, 0, GFE_C_HIERARCHICAL_RELATION_TOP_DOWN, 0.18, 1},
            };
            gfe_c_hierarchical_chain_spec_view chain_spec{
                chain_levels.data(),
                chain_levels.size(),
                chain_edges.data(),
                chain_edges.size(),
            };

            char errbuf6[256] = {0};
            const int chain_valid_rc =
                gfe_c_validate_hierarchical_chain_spec(&chain_spec, errbuf6, sizeof(errbuf6));
            if (chain_valid_rc != GFE_C_STATUS_OK) {
                std::cerr << "gfe_c_validate_hierarchical_chain_spec failed: " << errbuf6 << "\n";
                return 1;
            }

            auto hier_cfg = gfe_c_hierarchical_default_runtime_config();
            hier_cfg.steps = 18;
            hier_cfg.sample_stride = 6;
            hier_cfg.strict_finite = 1;

            std::vector<double> t_buf(4, 0.0);
            std::vector<std::size_t> state_dim_buf(4, 0);
            std::vector<std::size_t> memory_channels_buf(4, 0);
            std::vector<double> u0_buf(4, 0.0);
            std::vector<double> chi0_buf(4, 0.0);
            std::vector<double> w0_buf(4, 0.0);
            std::vector<double> gamma0_buf(4, 0.0);
            std::vector<double> mcap_buf(4, 0.0);
            std::vector<double> mscale_buf(4, 0.0);
            std::vector<double> hmem_buf(4, 0.0);
            std::vector<double> deff_buf(4, 0.0);
            std::size_t t_size = 0;
            std::size_t state_dim_size = 0;
            std::size_t memory_channels_size = 0;
            std::size_t u0_size = 0;
            std::size_t chi0_size = 0;
            std::size_t w0_size = 0;
            std::size_t gamma0_size = 0;
            std::size_t mcap_size = 0;
            std::size_t mscale_size = 0;
            std::size_t hmem_size = 0;
            std::size_t deff_size = 0;
            gfe_c_hierarchical_level_summary_view levels{
                t_buf.data(), t_buf.size(), &t_size,
                state_dim_buf.data(), state_dim_buf.size(), &state_dim_size,
                memory_channels_buf.data(), memory_channels_buf.size(), &memory_channels_size,
                u0_buf.data(), u0_buf.size(), &u0_size,
                chi0_buf.data(), chi0_buf.size(), &chi0_size,
                w0_buf.data(), w0_buf.size(), &w0_size,
                gamma0_buf.data(), gamma0_buf.size(), &gamma0_size,
                mcap_buf.data(), mcap_buf.size(), &mcap_size,
                mscale_buf.data(), mscale_buf.size(), &mscale_size,
                hmem_buf.data(), hmem_buf.size(), &hmem_size,
                deff_buf.data(), deff_buf.size(), &deff_size,
            };
            gfe_c_hierarchical_run_diagnostics diag{};
            const int chain_run_rc = gfe_c_hierarchical_run_chain_spec(
                &chain_spec, &hier_cfg, &levels, &diag, errbuf6, sizeof(errbuf6));
            if (chain_run_rc != GFE_C_STATUS_OK) {
                std::cerr << "gfe_c_hierarchical_run_chain_spec failed: " << errbuf6 << "\n";
                return 1;
            }
            if (!diag.all_finite || diag.level_count != 2 || t_size != 2 || deff_buf[0] <= 0.0) {
                std::cerr << "unexpected custom chain-spec run outputs\n";
                return 1;
            }

            std::vector<std::size_t> shift_level_idx(4, 0);
            std::vector<double> shift_delta_mcap(4, 0.0);
            std::vector<double> shift_delta_mscale(4, 0.0);
            std::vector<double> shift_delta_hmem(4, 0.0);
            std::vector<double> shift_delta_deff(4, 0.0);
            std::vector<int> shift_changed(4, 0);
            std::size_t shift_level_idx_size = 0;
            std::size_t shift_delta_mcap_size = 0;
            std::size_t shift_delta_mscale_size = 0;
            std::size_t shift_delta_hmem_size = 0;
            std::size_t shift_delta_deff_size = 0;
            std::size_t shift_changed_size = 0;
            gfe_c_hierarchical_kernel_shift_view kernel_shifts{
                shift_level_idx.data(), shift_level_idx.size(), &shift_level_idx_size,
                shift_delta_mcap.data(), shift_delta_mcap.size(), &shift_delta_mcap_size,
                shift_delta_mscale.data(), shift_delta_mscale.size(), &shift_delta_mscale_size,
                shift_delta_hmem.data(), shift_delta_hmem.size(), &shift_delta_hmem_size,
                shift_delta_deff.data(), shift_delta_deff.size(), &shift_delta_deff_size,
                shift_changed.data(), shift_changed.size(), &shift_changed_size,
            };

            std::vector<int> relation_type(8, 0);
            std::vector<std::size_t> relation_source(8, 0);
            std::vector<std::size_t> relation_target(8, 0);
            std::vector<double> relation_delta_mcap(8, 0.0);
            std::vector<double> relation_delta_mscale(8, 0.0);
            std::vector<double> relation_delta_hmem(8, 0.0);
            std::vector<double> relation_delta_deff(8, 0.0);
            std::vector<int> relation_target_mcap(8, 0);
            std::vector<int> relation_target_mscale(8, 0);
            std::size_t relation_type_size = 0;
            std::size_t relation_source_size = 0;
            std::size_t relation_target_size = 0;
            std::size_t relation_delta_mcap_size = 0;
            std::size_t relation_delta_mscale_size = 0;
            std::size_t relation_delta_hmem_size = 0;
            std::size_t relation_delta_deff_size = 0;
            std::size_t relation_target_mcap_size = 0;
            std::size_t relation_target_mscale_size = 0;
            gfe_c_hierarchical_relation_view relations{
                relation_type.data(), relation_type.size(), &relation_type_size,
                relation_source.data(), relation_source.size(), &relation_source_size,
                relation_target.data(), relation_target.size(), &relation_target_size,
                relation_delta_mcap.data(), relation_delta_mcap.size(), &relation_delta_mcap_size,
                relation_delta_mscale.data(), relation_delta_mscale.size(), &relation_delta_mscale_size,
                relation_delta_hmem.data(), relation_delta_hmem.size(), &relation_delta_hmem_size,
                relation_delta_deff.data(), relation_delta_deff.size(), &relation_delta_deff_size,
                relation_target_mcap.data(), relation_target_mcap.size(), &relation_target_mcap_size,
                relation_target_mscale.data(), relation_target_mscale.size(), &relation_target_mscale_size,
            };

            std::vector<int> warning_severity(8, 0);
            std::vector<int> warning_code(8, 0);
            std::vector<int> warning_metric(8, 0);
            std::vector<double> warning_observed(8, 0.0);
            std::vector<double> warning_threshold(8, 0.0);
            std::vector<std::size_t> warning_source(8, 0);
            std::vector<std::size_t> warning_target(8, 0);
            std::vector<int> warning_actionable(8, 0);
            std::size_t warning_severity_size = 0;
            std::size_t warning_code_size = 0;
            std::size_t warning_metric_size = 0;
            std::size_t warning_observed_size = 0;
            std::size_t warning_threshold_size = 0;
            std::size_t warning_source_size = 0;
            std::size_t warning_target_size = 0;
            std::size_t warning_actionable_size = 0;
            gfe_c_hierarchical_warning_view warnings{
                warning_severity.data(), warning_severity.size(), &warning_severity_size,
                warning_code.data(), warning_code.size(), &warning_code_size,
                warning_metric.data(), warning_metric.size(), &warning_metric_size,
                warning_observed.data(), warning_observed.size(), &warning_observed_size,
                warning_threshold.data(), warning_threshold.size(), &warning_threshold_size,
                warning_source.data(), warning_source.size(), &warning_source_size,
                warning_target.data(), warning_target.size(), &warning_target_size,
                warning_actionable.data(), warning_actionable.size(), &warning_actionable_size,
            };
            gfe_c_hierarchical_cross_level_contract contract{};
            const int chain_cross_rc = gfe_c_hierarchical_get_cross_level_report_for_chain_spec(
                &chain_spec, &hier_cfg, &kernel_shifts, &relations, &warnings, &contract, errbuf6, sizeof(errbuf6));
            if (chain_cross_rc != GFE_C_STATUS_OK) {
                std::cerr << "gfe_c_hierarchical_get_cross_level_report_for_chain_spec failed: " << errbuf6 << "\n";
                return 1;
            }
            if (!contract.all_finite_pass || shift_level_idx_size != 2 || relation_type_size != 2) {
                std::cerr << "unexpected custom chain-spec cross-level outputs\n";
                return 1;
            }

            std::vector<std::size_t> renorm_input(4, 0);
            std::vector<std::size_t> renorm_bath(4, 0);
            std::vector<std::size_t> renorm_direct(4, 0);
            std::vector<int> renorm_diag(4, 0);
            std::vector<std::size_t> renorm_raw_channels(4, 0);
            std::vector<std::size_t> renorm_output_channels(4, 0);
            std::vector<std::size_t> renorm_pruned_channels(4, 0);
            std::vector<double> renorm_retained(4, 0.0);
            std::vector<double> renorm_fit_l2(4, 0.0);
            std::vector<double> renorm_pruned_fit_l2(4, 0.0);
            std::vector<double> renorm_fit_linf(4, 0.0);
            std::vector<double> renorm_pruned_fit_linf(4, 0.0);
            std::vector<double> renorm_delta_mcap(4, 0.0);
            std::vector<double> renorm_delta_mscale(4, 0.0);
            std::vector<double> renorm_delta_deff(4, 0.0);
            std::vector<int> renorm_consistency(4, 0);
            std::vector<int> renorm_mcap_pass(4, 0);
            std::vector<int> renorm_mscale_pass(4, 0);
            std::vector<int> renorm_deff_pass(4, 0);
            std::size_t renorm_input_size = 0;
            std::size_t renorm_bath_size = 0;
            std::size_t renorm_direct_size = 0;
            std::size_t renorm_diag_size = 0;
            std::size_t renorm_raw_channels_size = 0;
            std::size_t renorm_output_channels_size = 0;
            std::size_t renorm_pruned_channels_size = 0;
            std::size_t renorm_retained_size = 0;
            std::size_t renorm_fit_l2_size = 0;
            std::size_t renorm_pruned_fit_l2_size = 0;
            std::size_t renorm_fit_linf_size = 0;
            std::size_t renorm_pruned_fit_linf_size = 0;
            std::size_t renorm_delta_mcap_size = 0;
            std::size_t renorm_delta_mscale_size = 0;
            std::size_t renorm_delta_deff_size = 0;
            std::size_t renorm_consistency_size = 0;
            std::size_t renorm_mcap_pass_size = 0;
            std::size_t renorm_mscale_pass_size = 0;
            std::size_t renorm_deff_pass_size = 0;
            gfe_c_hierarchical_renorm_analysis_view renorm{
                renorm_input.data(), renorm_input.size(), &renorm_input_size,
                renorm_bath.data(), renorm_bath.size(), &renorm_bath_size,
                renorm_direct.data(), renorm_direct.size(), &renorm_direct_size,
                renorm_diag.data(), renorm_diag.size(), &renorm_diag_size,
                renorm_raw_channels.data(), renorm_raw_channels.size(), &renorm_raw_channels_size,
                renorm_output_channels.data(), renorm_output_channels.size(), &renorm_output_channels_size,
                renorm_pruned_channels.data(), renorm_pruned_channels.size(), &renorm_pruned_channels_size,
                renorm_retained.data(), renorm_retained.size(), &renorm_retained_size,
                renorm_fit_l2.data(), renorm_fit_l2.size(), &renorm_fit_l2_size,
                renorm_pruned_fit_l2.data(), renorm_pruned_fit_l2.size(), &renorm_pruned_fit_l2_size,
                renorm_fit_linf.data(), renorm_fit_linf.size(), &renorm_fit_linf_size,
                renorm_pruned_fit_linf.data(), renorm_pruned_fit_linf.size(), &renorm_pruned_fit_linf_size,
                renorm_delta_mcap.data(), renorm_delta_mcap.size(), &renorm_delta_mcap_size,
                renorm_delta_mscale.data(), renorm_delta_mscale.size(), &renorm_delta_mscale_size,
                renorm_delta_deff.data(), renorm_delta_deff.size(), &renorm_delta_deff_size,
                renorm_consistency.data(), renorm_consistency.size(), &renorm_consistency_size,
                renorm_mcap_pass.data(), renorm_mcap_pass.size(), &renorm_mcap_pass_size,
                renorm_mscale_pass.data(), renorm_mscale_pass.size(), &renorm_mscale_pass_size,
                renorm_deff_pass.data(), renorm_deff_pass.size(), &renorm_deff_pass_size,
            };
            const int chain_renorm_rc = gfe_c_hierarchical_get_renorm_report_for_chain_spec(
                &chain_spec, &renorm, errbuf6, sizeof(errbuf6));
            if (chain_renorm_rc != GFE_C_STATUS_OK) {
                std::cerr << "gfe_c_hierarchical_get_renorm_report_for_chain_spec failed: " << errbuf6 << "\n";
                return 1;
            }
            if (renorm_input_size != 1 || renorm_retained[0] < 0.0 || renorm_retained[0] > 1.0) {
                std::cerr << "unexpected custom chain-spec renorm outputs\n";
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "gfe_c_api_smoke_test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "gfe_c_api_smoke_test passed\n";
    return 0;
}
