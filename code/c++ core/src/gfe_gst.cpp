#include "gfe/gfe_gst.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace gfe {
namespace {

Mat identity(std::size_t n) {
    Mat I(n, Vec(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        I[i][i] = 1.0;
    }
    return I;
}

void validate_square(const Mat& A) {
    if (A.empty() || A.front().empty()) {
        throw std::invalid_argument("operator matrix must be non-empty");
    }
    const std::size_t n = A.size();
    for (const auto& row : A) {
        if (row.size() != n) {
            throw std::invalid_argument("operator matrix must be square");
        }
    }
}

void validate_symmetric(const Mat& A, double tol = 1e-10) {
    const std::size_t n = A.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (std::fabs(A[i][j] - A[j][i]) > tol) {
                throw std::invalid_argument("reference GST backend expects symmetric operator matrix");
            }
        }
    }
}

struct EigenDecomp {
    Vec eigenvalues;
    Mat eigenvectors; // columns are eigenvectors
};

EigenDecomp jacobi_eigendecompose(const Mat& A_in, int max_iters, double tol) {
    Mat A = A_in;
    const std::size_t n = A.size();
    Mat V = identity(n);

    for (int iter = 0; iter < max_iters; ++iter) {
        std::size_t p = 0;
        std::size_t q = 1;
        double max_off = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                const double v = std::fabs(A[i][j]);
                if (v > max_off) {
                    max_off = v;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_off < tol) {
            break;
        }

        const double app = A[p][p];
        const double aqq = A[q][q];
        const double apq = A[p][q];
        const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(phi);
        const double s = std::sin(phi);

        for (std::size_t k = 0; k < n; ++k) {
            if (k != p && k != q) {
                const double aik = A[k][p];
                const double akq = A[k][q];
                A[k][p] = c * aik - s * akq;
                A[p][k] = A[k][p];
                A[k][q] = s * aik + c * akq;
                A[q][k] = A[k][q];
            }
        }

        A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A[p][q] = 0.0;
        A[q][p] = 0.0;

        for (std::size_t k = 0; k < n; ++k) {
            const double vip = V[k][p];
            const double viq = V[k][q];
            V[k][p] = c * vip - s * viq;
            V[k][q] = s * vip + c * viq;
        }
    }

    Vec evals(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        evals[i] = A[i][i];
    }
    return EigenDecomp{evals, V};
}

double frob_orthogonality_error(const Mat& Q) {
    if (Q.empty()) {
        return 0.0;
    }
    const std::size_t n = Q.size();
    const std::size_t m = Q.front().size();
    double s = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            double dot = 0.0;
            for (std::size_t r = 0; r < n; ++r) {
                dot += Q[r][i] * Q[r][j];
            }
            const double target = (i == j) ? 1.0 : 0.0;
            const double e = dot - target;
            s += e * e;
        }
    }
    return std::sqrt(s);
}

Mat pick_columns(const Mat& V, const std::vector<std::size_t>& cols) {
    const std::size_t n = V.size();
    Mat out(n, Vec(cols.size(), 0.0));
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < cols.size(); ++c) {
            out[r][c] = V[r][cols[c]];
        }
    }
    return out;
}

double l2_norm(const Vec& x) {
    double s = 0.0;
    for (double v : x) {
        s += v * v;
    }
    return std::sqrt(s);
}

} // namespace

DenseLinearGSTAdapter::DenseLinearGSTAdapter(const DenseOperatorProvider* provider, const GSTOptions& options)
    : provider_(provider), options_(options) {
    if (!provider_) {
        throw std::invalid_argument("DenseLinearGSTAdapter requires a non-null operator provider");
    }
}

bool DenseLinearGSTAdapter::needs_update(double t) const {
    if (!metadata_.valid) {
        return true;
    }
    if (t < metadata_.valid_from || t > metadata_.valid_until) {
        return true;
    }
    if (options_.update_interval > 0.0 && std::fabs(t - metadata_.basis_time) >= options_.update_interval) {
        return true;
    }
    return false;
}

void DenseLinearGSTAdapter::update_basis(double t) {
    Mat A = provider_->operator_matrix(t);
    validate_square(A);
    validate_symmetric(A);

    const auto decomp = jacobi_eigendecompose(A, options_.jacobi_max_iters, options_.jacobi_tol);

    std::vector<std::size_t> order(decomp.eigenvalues.size());
    std::iota(order.begin(), order.end(), 0);
    if (options_.sort_descending) {
        std::sort(order.begin(), order.end(), [&](std::size_t i, std::size_t j) {
            return decomp.eigenvalues[i] > decomp.eigenvalues[j];
        });
    }

    std::vector<std::size_t> keep;
    keep.reserve(order.size());
    for (std::size_t idx : order) {
        if (std::fabs(decomp.eigenvalues[idx]) < options_.min_abs_eigenvalue) {
            continue;
        }
        keep.push_back(idx);
        if (options_.max_modes > 0 && keep.size() >= options_.max_modes) {
            break;
        }
    }

    if (keep.empty()) {
        throw std::runtime_error("GST mode selection kept zero modes");
    }

    basis_ = pick_columns(decomp.eigenvectors, keep);

    metadata_.eigenvalues.clear();
    metadata_.eigenvalues.reserve(keep.size());
    for (std::size_t idx : keep) {
        metadata_.eigenvalues.push_back(decomp.eigenvalues[idx]);
    }
    metadata_.basis_time = t;
    metadata_.valid_from = t;
    metadata_.valid_until = (options_.basis_validity_horizon > 0.0)
        ? (t + options_.basis_validity_horizon)
        : std::numeric_limits<double>::infinity();
    metadata_.orthogonality_error = frob_orthogonality_error(basis_);

    double max_abs_ev = 0.0;
    double min_abs_ev = std::numeric_limits<double>::infinity();
    for (double ev : metadata_.eigenvalues) {
        const double a = std::fabs(ev);
        max_abs_ev = std::max(max_abs_ev, a);
        min_abs_ev = std::min(min_abs_ev, a);
    }
    metadata_.condition_estimate = (min_abs_ev > 0.0) ? (max_abs_ev / min_abs_ev) : std::numeric_limits<double>::infinity();
    metadata_.valid = true;
    metadata_.note = "ok";
}

void DenseLinearGSTAdapter::ensure_basis(double t) const {
    if (!needs_update(t)) {
        return;
    }

    try {
        const_cast<DenseLinearGSTAdapter*>(this)->update_basis(t);
    } catch (const std::exception& e) {
        if (options_.failure_policy == GSTFailurePolicy::ThrowOnFailure) {
            throw;
        }
        metadata_.valid = false;
        metadata_.note = std::string("fallback identity: ") + e.what();
    }
}

State DenseLinearGSTAdapter::project_with_basis(const State& u) const {
    if (basis_.empty()) {
        return u;
    }
    if (basis_.size() != u.size()) {
        throw std::invalid_argument("GST basis/state dimension mismatch in project");
    }

    const std::size_t n_modes = basis_.front().size();
    Vec coeffs(n_modes, 0.0);
    for (std::size_t j = 0; j < n_modes; ++j) {
        double acc = 0.0;
        for (std::size_t i = 0; i < u.size(); ++i) {
            acc += basis_[i][j] * u[i];
        }
        coeffs[j] = acc;
    }
    return coeffs;
}

State DenseLinearGSTAdapter::reconstruct_with_basis(const State& coeffs) const {
    if (basis_.empty()) {
        return coeffs;
    }
    const std::size_t n = basis_.size();
    const std::size_t n_modes = basis_.front().size();
    if (coeffs.size() != n_modes) {
        throw std::invalid_argument("GST basis/coeff dimension mismatch in reconstruct");
    }

    Vec u(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double acc = 0.0;
        for (std::size_t j = 0; j < n_modes; ++j) {
            acc += basis_[i][j] * coeffs[j];
        }
        u[i] = acc;
    }
    return u;
}

State DenseLinearGSTAdapter::project(const State& u, double t) const {
    ensure_basis(t);

    if (!metadata_.valid && options_.failure_policy == GSTFailurePolicy::FallbackIdentity) {
        last_report_ = GSTProjectionReport{l2_norm(u), l2_norm(u), 0.0, true};
        return u;
    }

    const auto coeffs = project_with_basis(u);
    const auto recon = reconstruct_with_basis(coeffs);

    const double in_norm = l2_norm(u);
    Vec diff(u.size(), 0.0);
    for (std::size_t i = 0; i < u.size(); ++i) {
        diff[i] = recon[i] - u[i];
    }
    const double err = l2_norm(diff);
    const double rel = (in_norm > 1e-14) ? (err / in_norm) : err;
    last_report_ = GSTProjectionReport{in_norm, l2_norm(recon), rel, rel <= options_.reconstruction_tol};

    return coeffs;
}

State DenseLinearGSTAdapter::reconstruct(const State& coeffs, double t) const {
    ensure_basis(t);

    if (!metadata_.valid && options_.failure_policy == GSTFailurePolicy::FallbackIdentity) {
        return coeffs;
    }

    return reconstruct_with_basis(coeffs);
}

} // namespace gfe

