#include "gfe/gfe_gst.hpp"

#include <cmath>
#include <iostream>

namespace {

struct FixedSymmetricProvider final : public gfe::DenseOperatorProvider {
    gfe::Mat operator_matrix(double /*t*/) const override {
        return gfe::Mat{{2.0, 1.0}, {1.0, 2.0}};
    }
};

struct SwitchingProvider final : public gfe::DenseOperatorProvider {
    gfe::Mat operator_matrix(double t) const override {
        if (t < 1.0) {
            return gfe::Mat{{3.0, 0.0}, {0.0, 1.0}};
        }
        return gfe::Mat{{1.5, 0.2}, {0.2, 2.5}};
    }
};

bool approx(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol;
}

} // namespace

int main() {
    using namespace gfe;

    try {
        FixedSymmetricProvider provider;
        GSTOptions opts;
        opts.max_modes = 2;
        opts.reconstruction_tol = 1e-8;
        DenseLinearGSTAdapter gst(&provider, opts);

        const State u{1.2, -0.7};
        const auto coeffs = gst.project(u, 0.0);
        const auto u_rec = gst.reconstruct(coeffs, 0.0);

        if (coeffs.size() != 2 || u_rec.size() != 2) {
            std::cerr << "GST shape mismatch\n";
            return 1;
        }

        double err = 0.0;
        double norm = 0.0;
        for (std::size_t i = 0; i < u.size(); ++i) {
            const double d = u_rec[i] - u[i];
            err += d * d;
            norm += u[i] * u[i];
        }
        err = std::sqrt(err);
        norm = std::sqrt(norm);

        if (!(err / norm < 1e-8)) {
            std::cerr << "GST reconstruction error too large\n";
            return 1;
        }

        if (!gst.last_report().within_tolerance) {
            std::cerr << "GST report tolerance should pass\n";
            return 1;
        }

        if (gst.metadata().eigenvalues.size() != 2) {
            std::cerr << "GST eigen metadata mismatch\n";
            return 1;
        }

        // Mode selection check.
        GSTOptions trunc_opts;
        trunc_opts.max_modes = 1;
        DenseLinearGSTAdapter gst_trunc(&provider, trunc_opts);
        const auto coeffs_trunc = gst_trunc.project(u, 0.0);
        if (coeffs_trunc.size() != 1 || gst_trunc.metadata().eigenvalues.size() != 1) {
            std::cerr << "GST mode truncation failed\n";
            return 1;
        }

        // Basis lifecycle/update policy check.
        SwitchingProvider switching;
        GSTOptions lifecycle_opts;
        lifecycle_opts.update_interval = 0.5;
        DenseLinearGSTAdapter gst_dyn(&switching, lifecycle_opts);
        (void)gst_dyn.project(State{1.0, 0.0}, 0.0);
        const double t0 = gst_dyn.metadata().basis_time;
        (void)gst_dyn.project(State{1.0, 0.0}, 0.2);
        if (!approx(gst_dyn.metadata().basis_time, t0, 1e-12)) {
            std::cerr << "GST should not update before interval\n";
            return 1;
        }
        (void)gst_dyn.project(State{1.0, 0.0}, 0.6);
        if (!(gst_dyn.metadata().basis_time > t0)) {
            std::cerr << "GST should update after interval\n";
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "gst_smoke_test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "gst_smoke_test passed\n";
    return 0;
}

