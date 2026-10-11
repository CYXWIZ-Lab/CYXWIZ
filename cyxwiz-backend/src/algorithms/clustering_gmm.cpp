// Gaussian mixture by EM on ArrayFire (TOFIX140), following scikit-learn's
// GaussianMixture: log-space E-step through Cholesky factors, reg_covar on every
// covariance, "full" / "tied" / "diag" / "spherical", convergence on the mean
// per-sample log-likelihood, BIC / AIC from its parameter count.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/clustering.h"
#include "analysis_device.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace cyxwiz {

using namespace analysis_detail;

namespace {

#ifdef CYXWIZ_HAS_ARRAYFIRE

constexpr double kRegCovar = 1e-6;

int CovarianceParameters(const std::string& type, int k, int d) {
    if (type == "full") return k * d * (d + 1) / 2;
    if (type == "tied") return d * (d + 1) / 2;
    if (type == "diag") return k * d;
    return k;  // spherical
}

struct Mixture {
    af::array means;                 // [k, d]
    std::vector<af::array> covs;     // k matrices (one for tied), d x d
    af::array log_weights;           // [1, k]
};

const af::array& CovarianceOf(const Mixture& m, int component) {
    return m.covs.size() == 1 ? m.covs[0] : m.covs[static_cast<size_t>(component)];
}

// log N(x | mean, cov) for every row of x: [n, 1].
af::array LogGaussian(const af::array& x, const af::array& mean, const af::array& cov) {
    const dim_t n = x.dims(0), d = x.dims(1);
    af::array lower;
    if (af::cholesky(lower, cov, /*is_upper=*/false) != 0 ||
        !af::allTrue<bool>(af::isNaN(lower) == 0 && af::isInf(lower) == 0)) {
        throw Refusal("A covariance is not positive definite (too few points for a component; "
                      "try fewer components or a diag / spherical covariance)");
    }
    const af::array diff = x - af::tile(mean, static_cast<unsigned>(n), 1);
    const af::array whitened = af::solve(lower, diff.T(), AF_MAT_LOWER);  // [d, n]
    const af::array mahalanobis = af::sum(whitened * whitened, 0).T();
    const double log_det = 2.0 * af::sum<double>(af::log(af::diag(lower)));
    return -0.5 * (static_cast<double>(d) * std::log(2.0 * 3.14159265358979323846) + log_det + mahalanobis);
}

// Responsibilities [n, k] and the mean per-sample log-likelihood.
std::pair<af::array, double> EStep(const af::array& x, const Mixture& m) {
    const dim_t n = x.dims(0), k = m.means.dims(0);
    af::array log_prob = af::constant(0.0, n, k, x.type());
    for (int c = 0; c < static_cast<int>(k); ++c) {
        log_prob(af::span, c) = LogGaussian(x, m.means(c, af::span), CovarianceOf(m, c));
    }
    log_prob += af::tile(m.log_weights, static_cast<unsigned>(n), 1);
    const af::array peak = af::max(log_prob, 1);
    const af::array log_norm = peak + af::log(af::sum(af::exp(log_prob - af::tile(peak, 1, static_cast<unsigned>(k))), 1));
    const af::array resp = af::exp(log_prob - af::tile(log_norm, 1, static_cast<unsigned>(k)));
    return {resp, af::mean<double>(log_norm)};
}

Mixture MStep(const af::array& x, const af::array& resp, const std::string& type) {
    const dim_t n = x.dims(0), d = x.dims(1), k = resp.dims(1);
    const af::array eye = af::identity(d, d, x.type());
    const af::array nk = af::sum(resp, 0) + 10.0 * std::numeric_limits<double>::epsilon();  // [1, k]
    const af::array nk_col = af::tile(nk.T(), 1, static_cast<unsigned>(d));                  // [k, d]
    Mixture m;
    m.log_weights = af::log(nk / static_cast<double>(n));
    m.means = af::matmul(resp, x, AF_MAT_TRANS, AF_MAT_NONE) / nk_col;
    m.means.eval();
    if (type == "tied") {
        const af::array avg_x2 = af::matmul(x, x, AF_MAT_TRANS, AF_MAT_NONE);
        const af::array avg_means2 = af::matmul(nk_col * m.means, m.means, AF_MAT_TRANS, AF_MAT_NONE);
        m.covs.push_back((avg_x2 - avg_means2) / af::sum<double>(nk) + kRegCovar * eye);
    } else if (type == "full") {
        for (int c = 0; c < static_cast<int>(k); ++c) {
            const af::array diff = x - af::tile(m.means(c, af::span), static_cast<unsigned>(n), 1);
            const af::array weighted = diff * af::tile(resp(af::span, c), 1, static_cast<unsigned>(d));
            const double count = af::sum<double>(nk(c));
            m.covs.push_back(af::matmul(weighted, diff, AF_MAT_TRANS, AF_MAT_NONE) / count + kRegCovar * eye);
        }
    } else {  // diag, spherical: per-feature variances [k, d]
        const af::array avg_x2 = af::matmul(resp, x * x, AF_MAT_TRANS, AF_MAT_NONE) / nk_col;
        af::array variances = avg_x2 - m.means * m.means + kRegCovar;
        if (type == "spherical") variances = af::tile(af::mean(variances, 1), 1, static_cast<unsigned>(d));
        for (int c = 0; c < static_cast<int>(k); ++c) {
            m.covs.push_back(af::diag(variances(c, af::span).T(), 0, false));
        }
    }
    for (auto& cov : m.covs) cov.eval();
    return m;
}

struct EmRun {
    Mixture mixture;
    af::array resp;
    double mean_ll = -std::numeric_limits<double>::infinity();
    int iterations = 0;
    bool converged = false;
};

EmRun RunEm(const af::array& x, int k, const std::string& type, int max_iter, double tol, af::randomEngine& engine,
            const std::function<void(int, double)>& progress) {
    const dim_t n = x.dims(0);
    // Start: hard assignment to k-means++ seeds, then an M-step.
    af::array nearest, labels;
    af::min(nearest, labels, SquaredDistances(x, KMeansPlusPlus(x, k, engine)), 1);
    const af::array ids = af::tile(af::range(af::dim4(1, k), 1, u32), static_cast<unsigned>(n), 1);
    EmRun run;
    run.mixture = MStep(x, (af::tile(labels, 1, static_cast<unsigned>(k)) == ids).as(x.type()), type);
    double bound = -std::numeric_limits<double>::infinity();
    for (int iter = 0; iter < max_iter; ++iter) {
        const double previous = bound;
        const auto [resp, mean_ll] = EStep(x, run.mixture);
        run.mixture = MStep(x, resp, type);
        bound = mean_ll;
        run.iterations = iter + 1;
        if (progress) progress(iter + 1, mean_ll * static_cast<double>(n));
        if (std::abs(bound - previous) < tol) {
            run.converged = true;
            break;
        }
    }
    // Final E-step: responsibilities and likelihood of the returned parameters.
    const auto [resp, mean_ll] = EStep(x, run.mixture);
    run.resp = resp;
    run.mean_ll = mean_ll;
    return run;
}

#endif

}  // namespace

GMMResult Clustering::GMM(
    const std::vector<std::vector<double>>& data,
    int n_components,
    const std::string& covariance_type,
    int max_iter,
    double tol,
    int n_init,
    unsigned int seed,
    std::function<void(int, double)> progress_callback
) {
    GMMResult result;
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        result.error_message = refusal;
        return result;
    }
    const int n = static_cast<int>(data.size()), d = static_cast<int>(data[0].size());
    if (n_components < 1 || n_components > n) {
        result.error_message = "Number of components must be between 1 and the number of samples";
        return result;
    }
    if (covariance_type != "full" && covariance_type != "tied" && covariance_type != "diag" &&
        covariance_type != "spherical") {
        result.error_message = "Unknown covariance type '" + covariance_type + "' (full, tied, diag or spherical)";
        return result;
    }
    if (max_iter < 1 || n_init < 1 || !(tol >= 0.0)) {
        result.error_message = "max_iter and n_init must be at least 1, and tol non-negative";
        return result;
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array x = MatrixToDevice(data);
        af::randomEngine engine = SeededEngine(seed);
        EmRun best;
        for (int run = 0; run < n_init; ++run) {
            EmRun current = RunEm(x, n_components, covariance_type, max_iter, tol, engine,
                                  run == 0 ? progress_callback : std::function<void(int, double)>{});
            if (current.mean_ll > best.mean_ll) best = std::move(current);
        }
        constexpr const char* kOperation = "Clustering::GMM";
        af::array peak, labels;
        af::max(peak, labels, best.resp, 1);
        result.labels = LabelsFromDevice(labels, kOperation);
        result.responsibilities = MatrixFromDevice(best.resp, kOperation);
        result.means = MatrixFromDevice(best.mixture.means, kOperation);
        result.weights = DeviceToHostValues(af::exp(best.mixture.log_weights), kOperation);
        for (int c = 0; c < n_components; ++c) {
            result.covariances.push_back(MatrixFromDevice(CovarianceOf(best.mixture, c), kOperation));
        }
        result.log_likelihood = best.mean_ll * n;
        const int parameters = n_components * d + CovarianceParameters(covariance_type, n_components, d) + n_components - 1;
        result.bic = -2.0 * result.log_likelihood + parameters * std::log(static_cast<double>(n));
        result.aic = -2.0 * result.log_likelihood + 2.0 * parameters;
        result.n_components = n_components;
        result.n_iterations = best.iterations;
        result.converged = best.converged;
        result.success = true;
    } catch (const Refusal& refusal) {
        result = GMMResult{};
        result.error_message = refusal.what();
    } catch (const af::exception& e) {
        result = GMMResult{};
        result.error_message = DeviceError("Clustering::GMM", e);
    }
#else
    (void)n;
    (void)d;
    (void)progress_callback;
    (void)seed;
    result.error_message = NoArrayFire("Clustering::GMM");
#endif
    return result;
}

} // namespace cyxwiz
