#pragma once

#include <utility>
#include <vector>

#include "gfe/gfe_backend.hpp"

namespace gfe {

using Vec = std::vector<double>;
using Mat = std::vector<Vec>; // row-major: Mat[row][col]

struct SpectralUnits {
    double Mcap;
    double Mscale;
    double Mres;
    double Hmem;
    double Hnorm;
    double Deff;
};

enum class DeffPolicy {
    EffectiveRankOnly,      // Deff = exp(Hmem)  (legacy)
    PaperEffectiveDimension // Deff = L * exp(Hmem)  (Spectral Units paper)
};

enum class McapPolicy {
    LegacyWeightedTimescale, // Mcap = sum(w/gamma) / sum(w)  (legacy)
    PaperMeanDepth           // Mcap = sum(w/gamma^2) / sum(w/gamma) (Spectral Units paper)
};

enum class MresPolicy {
    ReturnNaN,       // Mres = NaN when Mscale == 0
    ReturnInfinity,  // Mres = +inf when Mscale == 0
    ThrowOnZeroScale // throw std::invalid_argument when Mscale == 0
};

struct SpectralUnitOptions {
    DeffPolicy deff_policy = DeffPolicy::PaperEffectiveDimension;
    McapPolicy mcap_policy = McapPolicy::PaperMeanDepth;
    MresPolicy mres_policy = MresPolicy::ReturnNaN;
    double mscale_zero_tol = 1e-12;
};

struct FitResult {
    Vec gamma_fit;
    Vec w_fit;
    Vec fit;
};

enum class FitBackend {
    NnlsProjectedGradient,
    Prony
};

struct PronyOptions {
    int order = 6;
    double ridge = 1e-10;
    double root_imag_tol = 1e-6;
    double root_min = 1e-9;
    double root_max = 1.0 - 1e-9;
    bool enforce_nonnegative_weights = true;
    int max_root_iters = 300;
    double root_tol = 1e-10;
};

struct SoeFitOptions {
    FitBackend backend = FitBackend::NnlsProjectedGradient;
    int n_basis = 15;
    double gamma_min = 1e-2;
    double gamma_max = 10.0;
    double threshold = 0.01;
    double lr = 1e-3;
    int iters = 4000;
    PronyOptions prony;
};

Mat exponential_basis(const Vec& gamma, const Vec& t);
Mat design_matrix(const Vec& gamma, const Vec& t);
Vec logspace_gamma_basis(int n_basis, double gamma_min, double gamma_max);

Vec nnls_pg(const Mat& A, const Vec& b, double lr = 1e-3, int iters = 4000, const Vec* w0 = nullptr);
FitResult fit_soe_on_basis(
    const Vec& t,
    const Vec& data,
    const Vec& gamma_basis,
    double threshold = 0.01,
    double lr = 1e-3,
    int iters = 4000);
FitResult fit_soe_prony(const Vec& t, const Vec& data, const PronyOptions& options = {});
FitResult fit_soe_kernel(const Vec& t, const Vec& data, const SoeFitOptions& options);
FitResult fit_soe_kernel(
    const Vec& t,
    const Vec& data,
    const SoeFitOptions& options,
    const BackendConfig& backend_cfg);

FitResult fit_soe_kernel(
    const Vec& t,
    const Vec& data,
    int n_basis = 15,
    double gamma_min = 1e-2,
    double gamma_max = 10.0,
    double threshold = 0.01,
    double lr = 1e-3,
    int iters = 4000);

FitResult fit_soe_kernel(
    const Vec& t,
    const Vec& data,
    int n_basis,
    double gamma_min,
    double gamma_max,
    double threshold,
    double lr,
    int iters,
    const BackendConfig& backend_cfg);

SpectralUnits spectral_units(const Vec& w, const Vec& gamma, const SpectralUnitOptions& options = {});
double memory_capacity(const Vec& w, const Vec& gamma);
double spectral_entropy(const Vec& w);
double effective_dimension(const Vec& w);

void validate_decay_ordering(const Vec& gamma);
Vec pack_memory_params(const Vec& gamma, const Vec& w);
std::pair<Vec, Vec> unpack_memory_params(const Vec& theta);

std::pair<Vec, Vec> create_memory_channels(
    int L,
    std::pair<double, double> gamma_range = {1e-2, 1.0},
    const Vec* w_init = nullptr);

} // namespace gfe
