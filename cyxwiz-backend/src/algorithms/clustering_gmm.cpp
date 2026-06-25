// Prevent Windows min/max macros from interfering with std::numeric_limits and af::max/min
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "cyxwiz/clustering.h"
#include "arrayfire_backend_utils.h"
#include <spdlog/spdlog.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

// Ensure Windows min/max macros are undefined
#ifdef _WIN32
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

namespace cyxwiz {

#ifdef CYXWIZ_HAS_ARRAYFIRE

static std::string BuildClusteringContext(
    const std::vector<std::vector<double>>& data,
    const std::string& extra = {})
{
    std::string context = "samples=" + std::to_string(data.size()) +
        "; features=" + std::to_string(data.empty() ? 0 : data[0].size());
    if (!extra.empty()) {
        context += "; ";
        context += extra;
    }
    return context;
}

static void LogClusteringBackendFailureOnce(
    const char* operation_name,
    const char* error_message,
    const std::string& data_context)
{
    const BackendFallbackReason reason = ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string context = BuildArrayFireBackendFallbackContext(data_context);
    if (!ShouldLogArrayFireBackendFallbackOnce(operation_name, reason, context)) {
        return;
    }

    std::string message = std::string("ArrayFire ") +
        (operation_name ? operation_name : "clustering operation") +
        " failed (reason=" + BackendFallbackReasonName(reason) +
        "); clustering operation cannot complete on the current ArrayFire backend.";
    if (!context.empty()) {
        message += " Context: ";
        message += context;
        message += ".";
    }
    if (reason != BackendFallbackReason::CudaJitParamOverflow &&
        error_message != nullptr &&
        error_message[0] != '\0') {
        message += " Error: ";
        message += error_message;
    }
    spdlog::error("{}", message);
}

// ==================== GMM Implementation ====================

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

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    int n_samples = static_cast<int>(data.size());
    int n_features = static_cast<int>(data[0].size());

    if (n_components <= 0 || n_components > n_samples) {
        result.error_message = "Invalid number of components";
        return result;
    }

    try {
        af::array af_data = ToAfArray(data);

        double best_ll = -std::numeric_limits<double>::max();
        af::array best_means;
        std::vector<af::array> best_covs;
        af::array best_weights;
        af::array best_resp;
        int best_iters = 0;
        bool best_converged = false;

        for (int init_run = 0; init_run < n_init; ++init_run) {
            unsigned int run_seed = (seed == 0) ? 0 : seed + init_run;

            // Initialize parameters
            af::array means;
            std::vector<af::array> covariances;
            af::array weights;
            InitializeGMM(af_data, n_components, means, covariances, weights, covariance_type, run_seed);

            double prev_ll = -std::numeric_limits<double>::max();
            af::array responsibilities;
            int iterations = 0;
            bool converged = false;

            for (int iter = 0; iter < max_iter; ++iter) {
                // E-Step
                responsibilities = EStep(af_data, means, covariances, weights);

                // M-Step
                MStep(af_data, responsibilities, means, covariances, weights, covariance_type);

                // Compute log-likelihood
                double ll = ComputeLogLikelihood(af_data, means, covariances, weights);

                if (init_run == 0 && progress_callback) {
                    progress_callback(iter + 1, ll);
                }

                if (std::abs(ll - prev_ll) < tol) {
                    converged = true;
                    iterations = iter + 1;
                    break;
                }

                prev_ll = ll;
                iterations = iter + 1;
            }

            double final_ll = ComputeLogLikelihood(af_data, means, covariances, weights);

            if (final_ll > best_ll) {
                best_ll = final_ll;
                best_means = means;
                best_covs = covariances;
                best_weights = weights;
                best_resp = responsibilities;
                best_iters = iterations;
                best_converged = converged;
            }
        }

        // Convert results
        result.means = FromAfArray(best_means);

        // Convert responsibilities
        result.responsibilities.resize(n_samples);
        std::vector<double> resp_flat(n_samples * n_components);
        best_resp.eval();
        best_resp.host(resp_flat.data());
        for (int i = 0; i < n_samples; ++i) {
            result.responsibilities[i].resize(n_components);
            for (int j = 0; j < n_components; ++j) {
                result.responsibilities[i][j] = resp_flat[j * n_samples + i];
            }
        }

        // Hard labels
        af::array max_vals, label_indices;
        af::max(max_vals, label_indices, best_resp, 1);
        label_indices.eval();
        result.labels = AfArrayToIntVector(label_indices);

        // Weights
        result.weights = AfArrayToDoubleVector(best_weights);

        // Covariances
        result.covariances.resize(n_components);
        for (int k = 0; k < n_components; ++k) {
            result.covariances[k] = FromAfArray(best_covs[k]);
        }

        result.log_likelihood = best_ll;
        result.n_components = n_components;
        result.n_iterations = best_iters;
        result.converged = best_converged;

        // BIC and AIC
        int n_params = n_components * n_features + n_components * n_features * (n_features + 1) / 2 + n_components - 1;
        result.bic = -2 * best_ll + n_params * std::log(static_cast<double>(n_samples));
        result.aic = -2 * best_ll + 2 * n_params;

        result.success = true;

        spdlog::info("GMM completed: {} components, log-likelihood={:.4f}", n_components, best_ll);

    } catch (const af::exception& e) {
        result.error_message = std::string("ArrayFire error: ") + e.what();
        LogClusteringBackendFailureOnce(
            "Clustering::GMM",
            e.what(),
            BuildClusteringContext(
                data,
                "components=" + std::to_string(n_components) +
                "; covariance_type=" + covariance_type));
    }

    return result;
}

void Clustering::InitializeGMM(
    const af::array& data,
    int n_components,
    af::array& means,
    std::vector<af::array>& covariances,
    af::array& weights,
    const std::string& covariance_type,
    unsigned int seed
) {
    int n_samples = static_cast<int>(data.dims(0));
    int n_features = static_cast<int>(data.dims(1));

    // Initialize means using k-means++
    means = InitializeCentroidsKMeansPP(data, n_components, seed);

    // Initialize weights uniformly
    weights = af::constant(1.0 / n_components, n_components, f64);

    // Initialize covariances
    covariances.clear();
    af::array data_centered = data - af::tile(af::mean(data, 0), n_samples, 1);
    data_centered.eval();
    af::array global_cov = af::matmul(data_centered.T(), data_centered) / static_cast<double>(n_samples - 1);
    global_cov.eval();

    for (int k = 0; k < n_components; ++k) {
        if (covariance_type == "spherical") {
            af::array diag_cov = af::diag(global_cov);
            diag_cov.eval();
            double var = af::mean<double>(diag_cov);
            af::array covariance = af::identity(n_features, n_features, f64) * var;
            covariance.eval();
            covariances.push_back(covariance);
        } else if (covariance_type == "diag") {
            af::array diag_cov = af::diag(global_cov);
            diag_cov.eval();
            af::array covariance = af::diag(diag_cov, 0, false);
            covariance.eval();
            covariances.push_back(covariance);
        } else {
            covariances.push_back(global_cov.copy());
        }
    }
}

af::array Clustering::EStep(
    const af::array& data,
    const af::array& means,
    const std::vector<af::array>& covariances,
    const af::array& weights
) {
    int n_samples = static_cast<int>(data.dims(0));
    int n_components = static_cast<int>(means.dims(0));

    af::array log_probs = af::constant(0.0, n_samples, n_components, f64);

    for (int k = 0; k < n_components; ++k) {
        af::array pdf = GaussianPDF(data, means(k, af::span), covariances[k]);
        pdf.eval();
        log_probs(af::span, k) = af::log(pdf + 1e-300) + std::log(weights(k).scalar<double>());
    }
    log_probs.eval();

    // Log-sum-exp for numerical stability
    af::array max_log = af::max(log_probs, 1);
    max_log.eval();
    af::array log_probs_shifted = log_probs - af::tile(max_log, 1, n_components);
    log_probs_shifted.eval();
    af::array sum_exp = af::sum(af::exp(log_probs_shifted), 1);
    sum_exp.eval();
    af::array log_sum = max_log + af::log(sum_exp);
    log_sum.eval();

    // Responsibilities
    af::array responsibilities = af::exp(log_probs - af::tile(log_sum, 1, n_components));
    responsibilities.eval();

    return responsibilities;
}

void Clustering::MStep(
    const af::array& data,
    const af::array& responsibilities,
    af::array& means,
    std::vector<af::array>& covariances,
    af::array& weights,
    const std::string& covariance_type
) {
    int n_samples = static_cast<int>(data.dims(0));
    int n_features = static_cast<int>(data.dims(1));
    int n_components = static_cast<int>(means.dims(0));

    // Update weights
    af::array nk = af::sum(responsibilities, 0).T();  // [n_components x 1]
    nk.eval();
    weights = nk / static_cast<double>(n_samples);
    weights.eval();

    // Update means
    for (int k = 0; k < n_components; ++k) {
        af::array resp_k = responsibilities(af::span, k);  // [n_samples x 1]
        resp_k.eval();
        af::array weighted_sum = af::sum(data * af::tile(resp_k, 1, n_features), 0);
        weighted_sum.eval();
        means(k, af::span) = weighted_sum / nk(k).scalar<double>();
    }
    means.eval();

    // Update covariances
    for (int k = 0; k < n_components; ++k) {
        af::array resp_k = responsibilities(af::span, k);
        resp_k.eval();
        af::array mean_k = means(k, af::span);
        af::array diff = data - af::tile(mean_k, n_samples, 1);
        diff.eval();

        if (covariance_type == "spherical") {
            af::array weighted_sq = af::sum(diff * diff * af::tile(resp_k, 1, n_features), 0);
            weighted_sq.eval();
            double var = af::sum<double>(weighted_sq) / (nk(k).scalar<double>() * n_features);
            covariances[k] = af::identity(n_features, n_features, f64) * var;
            covariances[k].eval();
        } else if (covariance_type == "diag") {
            af::array weighted_sq = af::sum(diff * diff * af::tile(resp_k, 1, n_features), 0);
            weighted_sq.eval();
            af::array variances = weighted_sq / nk(k).scalar<double>();
            variances.eval();
            covariances[k] = af::diag(variances.T(), 0, false);
            covariances[k].eval();
        } else {
            af::array weighted_diff = diff * af::tile(af::sqrt(resp_k), 1, n_features);
            weighted_diff.eval();
            covariances[k] = af::matmul(weighted_diff.T(), weighted_diff) / nk(k).scalar<double>();
            covariances[k].eval();
            // Add regularization
            covariances[k] += af::identity(n_features, n_features, f64) * 1e-6;
            covariances[k].eval();
        }
    }
}

af::array Clustering::GaussianPDF(
    const af::array& data,
    const af::array& mean,
    const af::array& covariance
) {
    int n_samples = static_cast<int>(data.dims(0));
    int n_features = static_cast<int>(data.dims(1));

    af::array diff = data - af::tile(mean, n_samples, 1);
    diff.eval();

    // Compute (x - mu)^T @ Sigma^-1 @ (x - mu) for each sample
    af::array cov_inv = af::inverse(covariance);
    cov_inv.eval();
    af::array mahalanobis = af::sum(af::matmul(diff, cov_inv) * diff, 1);
    mahalanobis.eval();

    af::array covariance_diag = af::diag(covariance);
    covariance_diag.eval();
    af::array log_abs_diag = af::log(af::abs(covariance_diag));
    log_abs_diag.eval();
    double log_det = af::sum<double>(log_abs_diag);
    double log_norm = -0.5 * (n_features * std::log(2 * M_PI) + log_det);

    af::array pdf = af::exp(log_norm - 0.5 * mahalanobis);
    pdf.eval();
    return pdf;
}

double Clustering::ComputeLogLikelihood(
    const af::array& data,
    const af::array& means,
    const std::vector<af::array>& covariances,
    const af::array& weights
) {
    int n_samples = static_cast<int>(data.dims(0));
    int n_components = static_cast<int>(means.dims(0));

    af::array weighted_probs = af::constant(0.0, n_samples, f64);

    for (int k = 0; k < n_components; ++k) {
        af::array pdf = GaussianPDF(data, means(k, af::span), covariances[k]);
        pdf.eval();
        weighted_probs += weights(k).scalar<double>() * pdf;
    }
    weighted_probs.eval();

    af::array log_probs = af::log(weighted_probs + 1e-300);
    log_probs.eval();
    return af::sum<double>(log_probs);
}


#else  // No ArrayFire - CPU fallback stubs

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
    result.error_message = "ArrayFire not available";
    return result;
}


#endif  // CYXWIZ_HAS_ARRAYFIRE

} // namespace cyxwiz
