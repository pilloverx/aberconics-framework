#include "gfe/gfe_core.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
    double T = 200.0;
    double dt = 0.01;
    double theta = 1.0;
    double alpha = 0.1;
    double sigma = 1.0;
    int channels = 3;
    double sigma_ch = 0.5;
    int maxlag = 200;
    int seed = 42;
    bool robust = false;
    std::string csv_path;
};

void usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [--T val] [--dt val] [--theta val] [--alpha val] [--channels N] [--maxlag N] [--seed N] [--csv path] [--robust]\n"
        << "Defaults reproduce a compact OU-noise workflow in CLI.\n";
}

Config parse_args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto require_val = [&](const char* name) {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string(name) + " requires a value");
            }
            return std::string(argv[++i]);
        };

        if (a == "--help" || a == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (a == "--T") {
            c.T = std::stod(require_val("--T"));
        } else if (a == "--dt") {
            c.dt = std::stod(require_val("--dt"));
        } else if (a == "--theta") {
            c.theta = std::stod(require_val("--theta"));
        } else if (a == "--alpha") {
            c.alpha = std::stod(require_val("--alpha"));
        } else if (a == "--channels") {
            c.channels = std::stoi(require_val("--channels"));
        } else if (a == "--maxlag") {
            c.maxlag = std::stoi(require_val("--maxlag"));
        } else if (a == "--seed") {
            c.seed = std::stoi(require_val("--seed"));
        } else if (a == "--robust") {
            c.robust = true;
        } else if (a == "--csv") {
            c.csv_path = require_val("--csv");
        } else {
            throw std::runtime_error("Unknown arg: " + a);
        }
    }
    return c;
}

std::vector<double> autocorr_ts(const std::vector<double>& x, int maxlag) {
    const int n = static_cast<int>(x.size());
    maxlag = std::min(maxlag, n - 1);

    const double mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(n);
    std::vector<double> y(n, 0.0);
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] - mean;
    }

    std::vector<double> c(maxlag + 1, 0.0);
    for (int lag = 0; lag <= maxlag; ++lag) {
        double s = 0.0;
        for (int i = 0; i < n - lag; ++i) {
            s += y[i] * y[i + lag];
        }
        c[lag] = s / static_cast<double>(n - lag);
    }

    if (!std::isfinite(c[0]) || std::fabs(c[0]) < 1e-14) {
        return std::vector<double>(maxlag + 1, 0.0);
    }
    for (double& v : c) {
        v /= c[0];
    }
    return c;
}

gfe::Vec nnls_pg_report_style(const gfe::Mat& A, const gfe::Vec& b, double lr = 1e-3, int iters = 4000) {
    if (A.empty() || A.front().empty()) {
        throw std::runtime_error("A must be non-empty");
    }
    const std::size_t m = A.size();
    const std::size_t n = A.front().size();
    if (b.size() != m) {
        throw std::runtime_error("A and b size mismatch");
    }

    double max_col_abs_sum = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            s += std::fabs(A[i][j]);
        }
        if (s > max_col_abs_sum) {
            max_col_abs_sum = s;
        }
    }
    if (max_col_abs_sum <= 1e-14) {
        max_col_abs_sum = 1.0;
    }

    const double bsum = std::accumulate(b.begin(), b.end(), 0.0);
    gfe::Vec w(n, bsum / static_cast<double>(n) / max_col_abs_sum);

    gfe::Vec r(m, 0.0);
    gfe::Vec g(n, 0.0);
    for (int k = 0; k < iters; ++k) {
        for (std::size_t i = 0; i < m; ++i) {
            double aw = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                aw += A[i][j] * w[j];
            }
            r[i] = aw - b[i];
        }

        double gnorm2 = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                s += A[i][j] * r[i];
            }
            g[j] = s;
            gnorm2 += s * s;
        }

        for (std::size_t j = 0; j < n; ++j) {
            w[j] -= lr * g[j];
            if (w[j] < 0.0) {
                w[j] = 0.0;
            }
        }

        if (std::sqrt(gnorm2) < 1e-6) {
            break;
        }
    }

    return w;
}

std::vector<double> ou_white(const Config& c, std::mt19937_64& rng) {
    const int n = static_cast<int>(std::floor(c.T / c.dt)) + 1;
    const int burnin = c.robust ? std::max(1000, n / 10) : 0;
    std::vector<double> out(n, 0.0);
    std::normal_distribution<double> normal(0.0, 1.0);

    double x = 0.0;
    const double sdt = std::sqrt(c.dt);
    for (int i = 0; i < burnin; ++i) {
        const double xi = normal(rng);
        x += (-c.theta * x) * c.dt + c.sigma * sdt * xi;
    }
    for (int i = 0; i < n; ++i) {
        out[i] = x;
        const double xi = normal(rng);
        x += (-c.theta * x) * c.dt + c.sigma * sdt * xi;
    }
    return out;
}

std::vector<double> colored_ou(const Config& c, std::mt19937_64& rng) {
    const int n = static_cast<int>(std::floor(c.T / c.dt)) + 1;
    const int burnin = c.robust ? std::max(1000, n / 10) : 0;
    std::vector<double> out(n, 0.0);
    std::normal_distribution<double> normal(0.0, 1.0);

    double x = 0.0;
    double eta = 0.0;
    const double sdt = std::sqrt(c.dt);
    for (int i = 0; i < burnin; ++i) {
        const double xi = normal(rng);
        x += (-c.theta * x + eta) * c.dt;
        eta += (-c.alpha * eta) * c.dt + c.sigma * sdt * xi;
    }
    for (int i = 0; i < n; ++i) {
        out[i] = x;
        const double xi = normal(rng);
        x += (-c.theta * x + eta) * c.dt;
        eta += (-c.alpha * eta) * c.dt + c.sigma * sdt * xi;
    }
    return out;
}

std::vector<double> aberconics_memory(const Config& c, std::mt19937_64& rng, const gfe::Vec& gamma, const gfe::Vec& w) {
    const int n = static_cast<int>(std::floor(c.T / c.dt)) + 1;
    const int burnin = c.robust ? std::max(1000, n / 10) : 0;
    std::vector<double> out(n, 0.0);
    std::normal_distribution<double> normal(0.0, 1.0);

    double x = 0.0;
    std::vector<double> chi(gamma.size(), 0.0);
    const double sdt = std::sqrt(c.dt);

    for (int k = 0; k < burnin; ++k) {
        double coupling = 0.0;
        for (std::size_t i = 0; i < chi.size(); ++i) {
            coupling += w[i] * chi[i];
        }
        x += (-c.theta * x + coupling) * c.dt;
        for (std::size_t i = 0; i < chi.size(); ++i) {
            chi[i] += (-gamma[i] * chi[i]) * c.dt + c.sigma_ch * sdt * normal(rng);
        }
    }

    for (int k = 0; k < n; ++k) {
        out[k] = x;

        double coupling = 0.0;
        for (std::size_t i = 0; i < chi.size(); ++i) {
            coupling += w[i] * chi[i];
        }
        x += (-c.theta * x + coupling) * c.dt;

        for (std::size_t i = 0; i < chi.size(); ++i) {
            chi[i] += (-gamma[i] * chi[i]) * c.dt + c.sigma_ch * sdt * normal(rng);
        }
    }

    return out;
}

void write_csv(const std::string& path, const std::vector<std::pair<std::string, double>>& kv) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open csv path: " + path);
    }
    out << "metric,value\n";
    for (const auto& [k, v] : kv) {
        out << k << ',' << std::setprecision(17) << v << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Config c = parse_args(argc, argv);
        if (c.dt <= 0.0 || c.T <= 0.0 || c.channels <= 0) {
            throw std::runtime_error("invalid configuration (dt,T,channels must be positive)");
        }

        std::mt19937_64 rng(static_cast<std::mt19937_64::result_type>(c.seed));

        const auto [gamma_ab, w_ab] = gfe::create_memory_channels(c.channels);

        const auto x_white = ou_white(c, rng);
        const auto x_col = colored_ou(c, rng);
        const auto x_ab = aberconics_memory(c, rng, gamma_ab, w_ab);

        auto all_finite = [](const std::vector<double>& v) {
            for (double x : v) {
                if (!std::isfinite(x)) return false;
            }
            return true;
        };
        if (!all_finite(x_white) || !all_finite(x_col) || !all_finite(x_ab)) {
            throw std::runtime_error("non-finite values detected in simulation trajectory");
        }

        const int maxlag = std::min(c.maxlag, static_cast<int>(x_col.size()) - 1);
        const auto acf_white = autocorr_ts(x_white, maxlag);
        const auto acf_col = autocorr_ts(x_col, maxlag);
        const auto acf_ab = autocorr_ts(x_ab, maxlag);

        gfe::Vec t_acf(maxlag + 1, 0.0);
        for (int i = 0; i <= maxlag; ++i) {
            t_acf[static_cast<std::size_t>(i)] = c.dt * static_cast<double>(i);
        }

        // Build fitting basis (matching Julia notebook setup)
        const int n_basis = 15;
        gfe::Vec gamma_basis(static_cast<std::size_t>(n_basis), 0.0);
        const double g_lo = std::log10(1e-2);
        const double g_hi = std::log10(10.0);
        const double step = (g_hi - g_lo) / static_cast<double>(n_basis - 1);
        for (int i = 0; i < n_basis; ++i) {
            gamma_basis[static_cast<std::size_t>(i)] = std::pow(10.0, g_lo + step * static_cast<double>(i));
        }

        const auto A = gfe::design_matrix(gamma_basis, t_acf);
        gfe::Vec acf_target = acf_col;
        if (c.robust) {
            for (double& v : acf_target) {
                if (v < 0.0) {
                    v = 0.0;
                }
                if (v > 1.0) {
                    v = 1.0;
                }
            }
        }

        gfe::Vec w_full = c.robust
            ? gfe::nnls_pg(A, acf_target, 1e-2, 8000, nullptr)
            : nnls_pg_report_style(A, acf_target, 1e-3, 4000);

        double maxw = 0.0;
        for (double wi : w_full) {
            if (std::isfinite(wi) && wi > maxw) {
                maxw = wi;
            }
        }
        if (maxw <= 1e-14) {
            throw std::runtime_error("NNLS produced near-zero weights; adjust params/horizon");
        }

        gfe::Vec gamma_fit;
        gfe::Vec w_fit;
        gamma_fit.reserve(static_cast<std::size_t>(n_basis));
        w_fit.reserve(static_cast<std::size_t>(n_basis));
        for (int i = 0; i < n_basis; ++i) {
            if (w_full[static_cast<std::size_t>(i)] > 0.01 * maxw) {
                gamma_fit.push_back(gamma_basis[static_cast<std::size_t>(i)]);
                w_fit.push_back(w_full[static_cast<std::size_t>(i)]);
            }
        }
        if (w_fit.empty()) {
            if (c.robust) {
                // keep strongest mode if pruning removes all
                std::size_t best = 0;
                for (std::size_t i = 1; i < w_full.size(); ++i) {
                    if (w_full[i] > w_full[best]) best = i;
                }
                gamma_fit.push_back(gamma_basis[best]);
                w_fit.push_back(w_full[best]);
            } else {
                throw std::runtime_error("Pruning removed all fitted modes");
            }
        }

        // sort descending gamma (required by packing routines and consistency)
        std::vector<std::size_t> idx(gamma_fit.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return gamma_fit[a] > gamma_fit[b]; });
        gfe::Vec gamma_sorted;
        gfe::Vec w_sorted;
        gamma_sorted.reserve(idx.size());
        w_sorted.reserve(idx.size());
        for (std::size_t i : idx) {
            gamma_sorted.push_back(gamma_fit[i]);
            w_sorted.push_back(w_fit[i]);
        }
        gamma_fit = std::move(gamma_sorted);
        w_fit = std::move(w_sorted);

        gfe::Vec acf_fit(acf_target.size(), 0.0);
        for (std::size_t r = 0; r < acf_fit.size(); ++r) {
            double acc = 0.0;
            if (c.robust) {
                for (int i = 0; i < n_basis; ++i) {
                    acc += A[r][static_cast<std::size_t>(i)] * w_full[static_cast<std::size_t>(i)];
                }
            } else {
                for (std::size_t j = 0; j < gamma_fit.size(); ++j) {
                    acc += std::exp(-gamma_fit[j] * t_acf[r]) * w_fit[j];
                }
            }
            acf_fit[r] = acc;
        }

        const auto su = gfe::spectral_units(w_fit, gamma_fit); // paper-default
        gfe::SpectralUnitOptions su_legacy_opts;
        su_legacy_opts.deff_policy = gfe::DeffPolicy::EffectiveRankOnly;
        su_legacy_opts.mcap_policy = gfe::McapPolicy::LegacyWeightedTimescale;
        const auto su_legacy = gfe::spectral_units(w_fit, gamma_fit, su_legacy_opts);

        double l1 = 0.0;
        for (std::size_t i = 0; i < acf_fit.size(); ++i) {
            l1 += std::fabs(acf_fit[i] - acf_col[i]);
        }
        l1 /= static_cast<double>(acf_fit.size());

        std::cout << "OU Noise Experiment (C++/CLI)\n";
        std::cout << "=============================\n";
        std::cout << "n_samples: " << x_col.size() << "\n";
        std::cout << "dt: " << c.dt << ", T: " << c.T << "\n";
        std::cout << "channels (Aberconics): " << c.channels << "\n";
        std::cout << "maxlag: " << maxlag << "\n\n";

        std::cout << "ACF snapshots\n";
        const int lag50 = std::min(50, maxlag);
        std::cout << "  white   rho(0)=" << acf_white[0] << " rho(" << lag50 << ")=" << acf_white[lag50] << "\n";
        std::cout << "  colored rho(0)=" << acf_col[0] << " rho(" << lag50 << ")=" << acf_col[lag50] << "\n";
        std::cout << "  aber    rho(0)=" << acf_ab[0] << " rho(" << lag50 << ")=" << acf_ab[lag50] << "\n\n";

        std::cout << "SOE fit\n";
        std::cout << "  kept modes: " << w_fit.size() << "/15\n";
        std::cout << "  L1 error: " << l1 << "\n\n";

        std::cout << "Spectral units (from fitted colored-OU kernel)\n";
        std::cout << "  Mcap (paper/default)= " << su.Mcap << "\n";
        std::cout << "  Mcap (legacy)       = " << su_legacy.Mcap << "\n";
        std::cout << "  Mscale= " << su.Mscale << "\n";
        std::cout << "  Mres  = " << su.Mres << "\n";
        std::cout << "  Hmem  = " << su.Hmem << "\n";
        std::cout << "  Hnorm = " << su.Hnorm << "\n";
        std::cout << "  Deff (paper/default)= " << su.Deff << "\n";
        std::cout << "  Deff (legacy)       = " << su_legacy.Deff << "\n";

        if (!c.csv_path.empty()) {
            write_csv(c.csv_path,
                {
                    {"n_samples", static_cast<double>(x_col.size())},
                    {"dt", c.dt},
                    {"T", c.T},
                    {"channels", static_cast<double>(c.channels)},
                    {"maxlag", static_cast<double>(maxlag)},
                    {"acf_white_lag50", acf_white[lag50]},
                    {"acf_col_lag50", acf_col[lag50]},
                    {"acf_ab_lag50", acf_ab[lag50]},
                    {"fit_modes_kept", static_cast<double>(w_fit.size())},
                    {"fit_l1_error", l1},
                    {"Mcap_paper_default", su.Mcap},
                    {"Mcap_legacy", su_legacy.Mcap},
                    {"Mscale", su.Mscale},
                    {"Mres", su.Mres},
                    {"Hmem", su.Hmem},
                    {"Hnorm", su.Hnorm},
                    {"Deff_paper_default", su.Deff},
                    {"Deff_legacy", su_legacy.Deff}
                }
            );
            std::cout << "\nCSV written: " << c.csv_path << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
