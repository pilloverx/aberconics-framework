#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "gfe/gfe_core.hpp"
#include "gfe/gfe_dynamics.hpp"

namespace gfe {

enum class GSTFailurePolicy {
    ThrowOnFailure,
    FallbackIdentity
};

struct GSTOptions {
    std::size_t max_modes = 0; // 0 = keep all modes
    double min_abs_eigenvalue = 0.0;
    bool sort_descending = true;
    double update_interval = 0.0;       // 0 = no forced periodic updates
    double basis_validity_horizon = 0.0; // 0 = valid indefinitely unless forced update
    int jacobi_max_iters = 200;
    double jacobi_tol = 1e-12;
    double reconstruction_tol = 1e-6;
    GSTFailurePolicy failure_policy = GSTFailurePolicy::ThrowOnFailure;
};

struct GSTModalMetadata {
    Vec eigenvalues;
    double basis_time = 0.0;
    double valid_from = 0.0;
    double valid_until = 0.0;
    double orthogonality_error = 0.0;
    double condition_estimate = 1.0;
    bool valid = false;
    std::string note;
};

struct GSTProjectionReport {
    double input_l2 = 0.0;
    double reconstruction_l2 = 0.0;
    double relative_reconstruction_error = 0.0;
    bool within_tolerance = true;
};

struct DenseOperatorProvider {
    virtual Mat operator_matrix(double t) const = 0;
    virtual ~DenseOperatorProvider() = default;
};

class DenseLinearGSTAdapter final : public GSTAdapter {
public:
    DenseLinearGSTAdapter(const DenseOperatorProvider* provider, const GSTOptions& options = {});

    bool needs_update(double t) const;
    void update_basis(double t);

    State project(const State& u, double t) const override;
    State reconstruct(const State& coeffs, double t) const override;

    const GSTModalMetadata& metadata() const { return metadata_; }
    const GSTProjectionReport& last_report() const { return last_report_; }

private:
    void ensure_basis(double t) const;
    State project_with_basis(const State& u) const;
    State reconstruct_with_basis(const State& coeffs) const;

    const DenseOperatorProvider* provider_ = nullptr;
    GSTOptions options_;

    mutable Mat basis_; // columns are modes in physical-space coordinates
    mutable GSTModalMetadata metadata_;
    mutable GSTProjectionReport last_report_;
};

} // namespace gfe

