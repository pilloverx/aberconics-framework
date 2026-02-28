#include "gfe/gfe_core.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace gfe {

namespace {

void validate_matrix(const Mat& A) {
    if (A.empty()) {
        throw std::invalid_argument("A must not be empty");
    }
    const std::size_t cols = A.front().size();
    if (cols == 0) {
        throw std::invalid_argument("A must have at least one column");
    }
    for (const auto& row : A) {
        if (row.size() != cols) {
            throw std::invalid_argument("A rows must have consistent width");
        }
    }
}

Vec matvec(const Mat& A, const Vec& x) {
    validate_matrix(A);
    const std::size_t rows = A.size();
    const std::size_t cols = A.front().size();
    if (x.size() != cols) {
        throw std::invalid_argument("matvec dimension mismatch");
    }
    Vec y(rows, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        double acc = 0.0;
        for (std::size_t j = 0; j < cols; ++j) {
            acc += A[i][j] * x[j];
        }
        y[i] = acc;
    }
    return y;
}

Mat ata(const Mat& A) {
    validate_matrix(A);
    const std::size_t rows = A.size();
    const std::size_t cols = A.front().size();
    Mat out(cols, Vec(cols, 0.0));
    for (std::size_t i = 0; i < cols; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            double acc = 0.0;
            for (std::size_t r = 0; r < rows; ++r) {
                acc += A[r][i] * A[r][j];
            }
            out[i][j] = acc;
        }
    }
    return out;
}

Vec atb(const Mat& A, const Vec& b) {
    validate_matrix(A);
    const std::size_t rows = A.size();
    const std::size_t cols = A.front().size();
    if (b.size() != rows) {
        throw std::invalid_argument("A and b must have same number of rows");
    }
    Vec out(cols, 0.0);
    for (std::size_t c = 0; c < cols; ++c) {
        double acc = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            acc += A[r][c] * b[r];
        }
        out[c] = acc;
    }
    return out;
}

double norm2(const Vec& v) {
    double sum = 0.0;
    for (double x : v) {
        sum += x * x;
    }
    return std::sqrt(sum);
}

Vec matvec_square(const Mat& M, const Vec& x) {
    return matvec(M, x);
}

Vec solve_linear_system(Mat A, Vec b) {
    validate_matrix(A);
    const std::size_t n = A.size();
    if (A.front().size() != n) {
        throw std::invalid_argument("solve_linear_system requires square matrix");
    }
    if (b.size() != n) {
        throw std::invalid_argument("solve_linear_system dimension mismatch");
    }

    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        double pivot_abs = std::fabs(A[k][k]);
        for (std::size_t r = k + 1; r < n; ++r) {
            const double cand = std::fabs(A[r][k]);
            if (cand > pivot_abs) {
                pivot = r;
                pivot_abs = cand;
            }
        }
        if (pivot_abs < 1e-14) {
            throw std::invalid_argument("linear system is singular or ill-conditioned");
        }
        if (pivot != k) {
            std::swap(A[k], A[pivot]);
            std::swap(b[k], b[pivot]);
        }

        const double diag = A[k][k];
        for (std::size_t r = k + 1; r < n; ++r) {
            const double factor = A[r][k] / diag;
            A[r][k] = 0.0;
            for (std::size_t c = k + 1; c < n; ++c) {
                A[r][c] -= factor * A[k][c];
            }
            b[r] -= factor * b[k];
        }
    }

    Vec x(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double rhs = b[i];
        for (std::size_t c = i + 1; c < n; ++c) {
            rhs -= A[i][c] * x[c];
        }
        x[i] = rhs / A[i][i];
    }
    return x;
}

Vec least_squares_ridge(const Mat& A, const Vec& b, double ridge) {
    Mat ATA = ata(A);
    Vec ATb = atb(A, b);
    for (std::size_t i = 0; i < ATA.size(); ++i) {
        ATA[i][i] += ridge;
    }
    return solve_linear_system(std::move(ATA), std::move(ATb));
}

std::complex<double> polyval(const Vec& coeff_descending, const std::complex<double>& z) {
    std::complex<double> acc = 0.0;
    for (double c : coeff_descending) {
        acc = acc * z + c;
    }
    return acc;
}

std::vector<std::complex<double>> durand_kerner_roots(const Vec& coeff_descending, int max_iters, double tol) {
    const std::size_t degree = coeff_descending.size() - 1;
    if (degree < 1) {
        return {};
    }

    const double radius = 0.5;
    std::vector<std::complex<double>> roots(degree);
    const double pi = std::acos(-1.0);
    for (std::size_t i = 0; i < degree; ++i) {
        const double theta = (2.0 * pi * static_cast<double>(i)) / static_cast<double>(degree);
        roots[i] = std::polar(radius, theta);
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        double max_delta = 0.0;
        for (std::size_t i = 0; i < degree; ++i) {
            std::complex<double> denom = 1.0;
            for (std::size_t j = 0; j < degree; ++j) {
                if (j != i) {
                    denom *= (roots[i] - roots[j]);
                }
            }
            if (std::abs(denom) < 1e-18) {
                continue;
            }
            const std::complex<double> delta = polyval(coeff_descending, roots[i]) / denom;
            roots[i] -= delta;
            max_delta = std::max(max_delta, std::abs(delta));
        }
        if (max_delta < tol) {
            break;
        }
    }

    return roots;
}

FitResult finalize_fit(const Vec& gamma_basis, const Vec& w, const Vec& fit, double threshold) {
    const double max_w = *std::max_element(w.begin(), w.end());
    Vec gamma_fit;
    Vec w_fit;
    gamma_fit.reserve(w.size());
    w_fit.reserve(w.size());

    for (std::size_t i = 0; i < w.size(); ++i) {
        if (w[i] > threshold * max_w) {
            gamma_fit.push_back(gamma_basis[i]);
            w_fit.push_back(w[i]);
        }
    }

    std::vector<std::size_t> idx(gamma_fit.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        return gamma_fit[a] > gamma_fit[b];
    });

    Vec gamma_sorted;
    Vec w_sorted;
    gamma_sorted.reserve(idx.size());
    w_sorted.reserve(idx.size());
    for (std::size_t i : idx) {
        gamma_sorted.push_back(gamma_fit[i]);
        w_sorted.push_back(w_fit[i]);
    }

    return FitResult{gamma_sorted, w_sorted, fit};
}

double compute_mres(std::size_t n_modes, double mscale, const SpectralUnitOptions& options) {
    if (std::abs(mscale) > options.mscale_zero_tol) {
        return static_cast<double>(n_modes) / mscale;
    }

    switch (options.mres_policy) {
        case MresPolicy::ReturnNaN:
            return std::numeric_limits<double>::quiet_NaN();
        case MresPolicy::ReturnInfinity:
            return std::numeric_limits<double>::infinity();
        case MresPolicy::ThrowOnZeroScale:
            throw std::invalid_argument("Mscale is zero (or near-zero); Mres is undefined under ThrowOnZeroScale policy");
    }
    throw std::invalid_argument("Unknown MresPolicy");
}

double compute_deff(std::size_t n_modes, double Hmem, DeffPolicy policy) {
    switch (policy) {
        case DeffPolicy::EffectiveRankOnly:
            return std::exp(Hmem);
        case DeffPolicy::PaperEffectiveDimension:
            return static_cast<double>(n_modes) * std::exp(Hmem);
    }
    throw std::invalid_argument("Unknown DeffPolicy");
}

double compute_mcap(const Vec& w, const Vec& gamma, McapPolicy policy) {
    const double W = std::accumulate(w.begin(), w.end(), 0.0);
    if (W == 0.0) {
        throw std::invalid_argument("sum(w) must be non-zero");
    }

    if (policy == McapPolicy::LegacyWeightedTimescale) {
        double num = 0.0;
        for (std::size_t i = 0; i < w.size(); ++i) {
            num += w[i] / gamma[i];
        }
        return num / W;
    }

    if (policy == McapPolicy::PaperMeanDepth) {
        double num = 0.0;
        double den = 0.0;
        for (std::size_t i = 0; i < w.size(); ++i) {
            const double inv_g = 1.0 / gamma[i];
            num += w[i] * inv_g * inv_g;
            den += w[i] * inv_g;
        }
        if (den == 0.0) {
            throw std::invalid_argument("paper Mcap denominator is zero");
        }
        return num / den;
    }

    throw std::invalid_argument("Unknown McapPolicy");
}

FitResult fit_soe_kernel_cpu(const Vec& t, const Vec& data, const SoeFitOptions& options) {
    switch (options.backend) {
        case FitBackend::NnlsProjectedGradient: {
            const Vec gamma_basis = logspace_gamma_basis(options.n_basis, options.gamma_min, options.gamma_max);
            return fit_soe_on_basis(t, data, gamma_basis, options.threshold, options.lr, options.iters);
        }
        case FitBackend::Prony:
            return fit_soe_prony(t, data, options.prony);
    }
    throw std::invalid_argument("Unknown FitBackend");
}

} // namespace

Mat exponential_basis(const Vec& gamma, const Vec& t) {
    const std::size_t n_basis = gamma.size();
    const std::size_t n_samples = t.size();
    Mat phi(n_samples, Vec(n_basis, 0.0));
    for (std::size_t i = 0; i < n_basis; ++i) {
        for (std::size_t r = 0; r < n_samples; ++r) {
            phi[r][i] = std::exp(-gamma[i] * t[r]);
        }
    }
    return phi;
}

Mat design_matrix(const Vec& gamma, const Vec& t) {
    return exponential_basis(gamma, t);
}

Vec logspace_gamma_basis(int n_basis, double gamma_min, double gamma_max) {
    if (n_basis < 1) {
        throw std::invalid_argument("n_basis must be >= 1");
    }
    if (gamma_min <= 0.0 || gamma_max <= 0.0 || gamma_min > gamma_max) {
        throw std::invalid_argument("gamma range must satisfy 0 < gamma_min <= gamma_max");
    }

    Vec gamma_basis(static_cast<std::size_t>(n_basis));
    const double lo = std::log10(gamma_min);
    const double hi = std::log10(gamma_max);
    if (n_basis == 1) {
        gamma_basis[0] = std::pow(10.0, lo);
        return gamma_basis;
    }

    const double step = (hi - lo) / static_cast<double>(n_basis - 1);
    for (int i = 0; i < n_basis; ++i) {
        gamma_basis[static_cast<std::size_t>(i)] = std::pow(10.0, lo + step * static_cast<double>(i));
    }
    return gamma_basis;
}

Vec nnls_pg(const Mat& A, const Vec& b, double lr, int iters, const Vec* w0) {
    validate_matrix(A);
    const std::size_t n_samples = A.size();
    const std::size_t n_basis = A.front().size();
    if (b.size() != n_samples) {
        throw std::invalid_argument("A and b must have same number of rows");
    }

    Vec w = w0 ? *w0 : Vec(n_basis, 0.0);
    if (w.size() != n_basis) {
        throw std::invalid_argument("w0 length must equal number of A columns");
    }

    const Mat ATA = ata(A);
    const Vec ATb = atb(A, b);

    for (int iter = 0; iter < iters; ++iter) {
        const Vec ATAw = matvec_square(ATA, w);
        Vec grad(n_basis, 0.0);
        for (std::size_t i = 0; i < n_basis; ++i) {
            grad[i] = 2.0 * (ATAw[i] - ATb[i]);
        }

        Vec w_new(n_basis, 0.0);
        for (std::size_t i = 0; i < n_basis; ++i) {
            const double cand = w[i] - lr * grad[i];
            w_new[i] = cand > 0.0 ? cand : 0.0;
        }

        Vec delta(n_basis, 0.0);
        for (std::size_t i = 0; i < n_basis; ++i) {
            delta[i] = w_new[i] - w[i];
        }
        if (norm2(delta) < 1e-10) {
            w = std::move(w_new);
            break;
        }

        w = std::move(w_new);
    }

    return w;
}

FitResult fit_soe_on_basis(const Vec& t,
                           const Vec& data,
                           const Vec& gamma_basis,
                           double threshold,
                           double lr,
                           int iters) {
    if (t.size() != data.size()) {
        throw std::invalid_argument("t and data must have the same length");
    }
    if (gamma_basis.empty()) {
        throw std::invalid_argument("gamma_basis must be non-empty");
    }

    const Mat A = design_matrix(gamma_basis, t);
    const Vec w = nnls_pg(A, data, lr, iters, nullptr);
    const Vec fit = matvec(A, w);
    return finalize_fit(gamma_basis, w, fit, threshold);
}

FitResult fit_soe_prony(const Vec& t, const Vec& data, const PronyOptions& options) {
    if (t.size() != data.size()) {
        throw std::invalid_argument("t and data must have the same length");
    }
    if (t.size() < 4) {
        throw std::invalid_argument("Prony fit requires at least 4 time samples");
    }

    const double dt = t[1] - t[0];
    if (dt <= 0.0) {
        throw std::invalid_argument("t must be strictly increasing for Prony fit");
    }
    for (std::size_t i = 2; i < t.size(); ++i) {
        const double dti = t[i] - t[i - 1];
        if (std::fabs(dti - dt) > 1e-6 * std::max(1.0, std::fabs(dt))) {
            throw std::invalid_argument("Prony fit requires approximately uniform time spacing");
        }
    }

    const int order_cap = static_cast<int>(t.size()) / 2;
    const int m = std::max(1, std::min(options.order, order_cap));
    const int rows = static_cast<int>(t.size()) - m;
    if (rows <= m) {
        throw std::invalid_argument("Prony fit has insufficient data for requested order");
    }

    Mat T(static_cast<std::size_t>(rows), Vec(static_cast<std::size_t>(m), 0.0));
    Vec rhs(static_cast<std::size_t>(rows), 0.0);
    for (int r = 0; r < rows; ++r) {
        const int k = m + r;
        rhs[static_cast<std::size_t>(r)] = -data[static_cast<std::size_t>(k)];
        for (int j = 0; j < m; ++j) {
            T[static_cast<std::size_t>(r)][static_cast<std::size_t>(j)] =
                data[static_cast<std::size_t>(k - j - 1)];
        }
    }

    const Vec c = least_squares_ridge(T, rhs, options.ridge);

    Vec poly;
    poly.reserve(static_cast<std::size_t>(m + 1));
    poly.push_back(1.0);
    for (double ci : c) {
        poly.push_back(ci);
    }

    const auto roots = durand_kerner_roots(poly, options.max_root_iters, options.root_tol);

    Vec gamma_candidates;
    for (const auto& z : roots) {
        if (std::fabs(z.imag()) > options.root_imag_tol) {
            continue;
        }
        const double zr = z.real();
        if (zr <= options.root_min || zr >= options.root_max) {
            continue;
        }
        const double gamma = -std::log(zr) / dt;
        if (std::isfinite(gamma) && gamma > 0.0) {
            gamma_candidates.push_back(gamma);
        }
    }

    if (gamma_candidates.empty()) {
        throw std::invalid_argument("Prony fit did not recover valid decay rates");
    }

    std::sort(gamma_candidates.begin(), gamma_candidates.end(), std::greater<double>());
    gamma_candidates.erase(
        std::unique(gamma_candidates.begin(), gamma_candidates.end(), [](double a, double b) {
            return std::fabs(a - b) < 1e-8 * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
        }),
        gamma_candidates.end());

    const Mat A = design_matrix(gamma_candidates, t);
    Vec w = least_squares_ridge(A, data, options.ridge);
    if (options.enforce_nonnegative_weights) {
        for (double& wi : w) {
            wi = std::max(0.0, wi);
        }
    }

    const Vec fit = matvec(A, w);
    return finalize_fit(gamma_candidates, w, fit, 0.0);
}

FitResult fit_soe_kernel(const Vec& t, const Vec& data, const SoeFitOptions& options) {
    BackendConfig backend_cfg;
    backend_cfg.preferred = get_default_backend();
    backend_cfg.allow_fallback = true;
    backend_cfg.fallback = ComputeBackend::CpuReference;
    return fit_soe_kernel(t, data, options, backend_cfg);
}

FitResult fit_soe_kernel(const Vec& t, const Vec& data, const SoeFitOptions& options, const BackendConfig& backend_cfg) {
    const auto selection = select_backend(backend_cfg);
    switch (selection.resolved) {
        case ComputeBackend::CpuReference:
            return fit_soe_kernel_cpu(t, data, options);
        case ComputeBackend::CpuBlas:
        case ComputeBackend::GpuCuda:
        case ComputeBackend::GpuHip:
        case ComputeBackend::GpuSycl:
            break;
    }
    throw std::invalid_argument(std::string("Resolved backend is not implemented for fit_soe_kernel: ") +
                                backend_name(selection.resolved));
}

FitResult fit_soe_kernel(const Vec& t,
                         const Vec& data,
                         int n_basis,
                         double gamma_min,
                         double gamma_max,
                         double threshold,
                         double lr,
                         int iters) {
    SoeFitOptions options;
    options.backend = FitBackend::NnlsProjectedGradient;
    options.n_basis = n_basis;
    options.gamma_min = gamma_min;
    options.gamma_max = gamma_max;
    options.threshold = threshold;
    options.lr = lr;
    options.iters = iters;
    return fit_soe_kernel(t, data, options);
}

FitResult fit_soe_kernel(const Vec& t,
                         const Vec& data,
                         int n_basis,
                         double gamma_min,
                         double gamma_max,
                         double threshold,
                         double lr,
                         int iters,
                         const BackendConfig& backend_cfg) {
    SoeFitOptions options;
    options.backend = FitBackend::NnlsProjectedGradient;
    options.n_basis = n_basis;
    options.gamma_min = gamma_min;
    options.gamma_max = gamma_max;
    options.threshold = threshold;
    options.lr = lr;
    options.iters = iters;
    return fit_soe_kernel(t, data, options, backend_cfg);
}

SpectralUnits spectral_units(const Vec& w, const Vec& gamma, const SpectralUnitOptions& options) {
    if (w.size() != gamma.size() || w.empty()) {
        throw std::invalid_argument("w and gamma must have the same non-zero length");
    }

    const double W = std::accumulate(w.begin(), w.end(), 0.0);
    const double Mcap = compute_mcap(w, gamma, options.mcap_policy);

    const auto [gmin_it, gmax_it] = std::minmax_element(gamma.begin(), gamma.end());
    if (*gmin_it <= 0.0) {
        throw std::invalid_argument("gamma values must be strictly positive");
    }
    const double Mscale = std::log10((*gmax_it) / (*gmin_it));
    const double Mres = compute_mres(w.size(), Mscale, options);

    const double eps = std::numeric_limits<double>::epsilon();
    double Hmem = 0.0;
    for (double wi : w) {
        const double p = wi / W;
        Hmem -= p * std::log(p + eps);
    }

    const double Hnorm = Hmem / std::log(std::max<std::size_t>(w.size(), 2));
    const double Deff = compute_deff(w.size(), Hmem, options.deff_policy);

    return SpectralUnits{Mcap, Mscale, Mres, Hmem, Hnorm, Deff};
}

double memory_capacity(const Vec& w, const Vec& gamma) {
    if (w.size() != gamma.size() || w.empty()) {
        throw std::invalid_argument("w and gamma must have the same non-zero length");
    }
    double num = 0.0; // Σ w/gamma^2
    double den = 0.0; // Σ w/gamma
    for (std::size_t i = 0; i < w.size(); ++i) {
        const double inv_g = 1.0 / gamma[i];
        num += w[i] * inv_g * inv_g;
        den += w[i] * inv_g;
    }
    if (den == 0.0) {
        throw std::invalid_argument("memory_capacity denominator is zero");
    }
    return num / den;
}

double spectral_entropy(const Vec& w) {
    if (w.empty()) {
        throw std::invalid_argument("w must not be empty");
    }
    const double W = std::accumulate(w.begin(), w.end(), 0.0);
    const double eps = std::numeric_limits<double>::epsilon();
    double H = 0.0;
    for (double wi : w) {
        const double p = wi / W;
        H -= p * std::log(p + eps);
    }
    return H / std::log(std::max<std::size_t>(w.size(), 2));
}

double effective_dimension(const Vec& w) {
    if (w.empty()) {
        throw std::invalid_argument("w must not be empty");
    }
    const double W = std::accumulate(w.begin(), w.end(), 0.0);
    const double eps = std::numeric_limits<double>::epsilon();
    double H = 0.0;
    for (double wi : w) {
        const double p = wi / W;
        H -= p * std::log(p + eps);
    }
    return static_cast<double>(w.size()) * std::exp(H);
}

void validate_decay_ordering(const Vec& gamma) {
    if (!std::is_sorted(gamma.begin(), gamma.end(), std::greater<double>())) {
        throw std::invalid_argument("Decay rates gamma must be sorted in descending order");
    }
}

Vec pack_memory_params(const Vec& gamma, const Vec& w) {
    validate_decay_ordering(gamma);
    if (gamma.size() != w.size() || gamma.empty()) {
        throw std::invalid_argument("gamma and w must have same non-zero length");
    }

    const std::size_t L = gamma.size();
    Vec dgamma(L, 0.0);
    dgamma[L - 1] = std::max(gamma[L - 1], 1e-12);
    for (std::size_t i = L - 1; i-- > 0;) {
        dgamma[i] = std::max(gamma[i] - gamma[i + 1], 1e-12);
    }

    Vec theta;
    theta.reserve(2 * L);
    for (double d : dgamma) {
        theta.push_back(std::log(d));
    }
    for (double wi : w) {
        if (wi <= 0.0) {
            throw std::invalid_argument("weights must be strictly positive for log parametrization");
        }
        theta.push_back(std::log(wi));
    }
    return theta;
}

std::pair<Vec, Vec> unpack_memory_params(const Vec& theta) {
    if (theta.size() % 2 != 0 || theta.empty()) {
        throw std::invalid_argument("theta must have even non-zero length");
    }

    const std::size_t L = theta.size() / 2;
    Vec dgamma(L, 0.0);
    for (std::size_t i = 0; i < L; ++i) {
        dgamma[i] = std::exp(theta[i]);
    }

    Vec gamma(L, 0.0);
    gamma[L - 1] = dgamma[L - 1];
    for (std::size_t i = L - 1; i-- > 0;) {
        gamma[i] = gamma[i + 1] + dgamma[i];
    }

    Vec w(L, 0.0);
    for (std::size_t i = 0; i < L; ++i) {
        w[i] = std::exp(theta[L + i]);
    }

    return {gamma, w};
}

std::pair<Vec, Vec> create_memory_channels(int L, std::pair<double, double> gamma_range, const Vec* w_init) {
    if (L <= 0) {
        throw std::invalid_argument("L must be positive");
    }
    const double gamma_min = gamma_range.first;
    const double gamma_max = gamma_range.second;
    if (gamma_min <= 0.0 || gamma_max <= 0.0 || gamma_min > gamma_max) {
        throw std::invalid_argument("gamma_range must satisfy 0 < gamma_min <= gamma_max");
    }

    Vec gamma(static_cast<std::size_t>(L), 0.0);
    const double lo = std::log10(gamma_max);
    const double hi = std::log10(gamma_min);
    if (L == 1) {
        gamma[0] = std::pow(10.0, lo);
    } else {
        const double step = (hi - lo) / static_cast<double>(L - 1);
        for (int i = 0; i < L; ++i) {
            gamma[static_cast<std::size_t>(i)] = std::pow(10.0, lo + step * static_cast<double>(i));
        }
    }

    Vec w;
    if (w_init == nullptr) {
        w.assign(static_cast<std::size_t>(L), 1.0 / static_cast<double>(L));
    } else {
        if (w_init->size() != static_cast<std::size_t>(L)) {
            throw std::invalid_argument("w_init length must equal L");
        }
        w = *w_init;
    }

    return {gamma, w};
}

} // namespace gfe
