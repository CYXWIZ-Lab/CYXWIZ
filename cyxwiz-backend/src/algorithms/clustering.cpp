// Prevent Windows min/max macros from interfering with std::numeric_limits and af::max/min
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "cyxwiz/clustering.h"
#include <spdlog/spdlog.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <numeric>
#include <set>
#include <queue>

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

// ==================== ArrayFire Conversion Helpers ====================

af::array Clustering::ToAfArray(const std::vector<std::vector<double>>& data) {
    if (data.empty()) return af::array();

    int n_samples = static_cast<int>(data.size());
    int n_features = static_cast<int>(data[0].size());

    // Flatten data in column-major order for ArrayFire
    std::vector<double> flat_data(n_samples * n_features);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            flat_data[j * n_samples + i] = data[i][j];
        }
    }

    return af::array(n_samples, n_features, flat_data.data());
}

std::vector<std::vector<double>> Clustering::FromAfArray(const af::array& arr) {
    if (arr.isempty()) return {};

    int n_samples = static_cast<int>(arr.dims(0));
    int n_features = static_cast<int>(arr.dims(1));

    std::vector<double> flat_data(n_samples * n_features);
    arr.host(flat_data.data());

    std::vector<std::vector<double>> result(n_samples, std::vector<double>(n_features));
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            result[i][j] = flat_data[j * n_samples + i];
        }
    }

    return result;
}

std::vector<int> Clustering::AfArrayToIntVector(const af::array& arr) {
    if (arr.isempty()) return {};

    int n = static_cast<int>(arr.elements());
    std::vector<int> result(n);

    // Convert to int array on host
    af::array int_arr = arr.as(s32);
    int_arr.host(result.data());

    return result;
}

std::vector<double> Clustering::AfArrayToDoubleVector(const af::array& arr) {
    if (arr.isempty()) return {};

    int n = static_cast<int>(arr.elements());
    std::vector<double> result(n);

    af::array double_arr = arr.as(f64);
    double_arr.host(result.data());

    return result;
}

// ==================== GPU Distance Functions ====================

af::array Clustering::ComputeEuclideanDistanceMatrix(const af::array& data) {
    // data: [n_samples x n_features]
    // Output: [n_samples x n_samples] distance matrix

    int n = static_cast<int>(data.dims(0));

    // ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
    af::array sq_norms = af::sum(data * data, 1);  // [n x 1]

    // Expand for broadcasting
    af::array sq_norms_row = af::tile(sq_norms, 1, n);          // [n x n]
    af::array sq_norms_col = af::tile(sq_norms.T(), n, 1);      // [n x n]

    // Compute dot products: data @ data.T
    af::array dot_products = af::matmul(data, data.T());  // [n x n]

    // Squared distances
    af::array sq_distances = sq_norms_row + sq_norms_col - 2.0 * dot_products;

    // Clamp negative values (numerical errors) and take sqrt
    sq_distances = af::max(sq_distances, 0.0);
    return af::sqrt(sq_distances);
}

af::array Clustering::ComputeManhattanDistanceMatrix(const af::array& data) {
    int n = static_cast<int>(data.dims(0));
    int d = static_cast<int>(data.dims(1));

    // Expand data for pairwise computation
    // data_i: [n x 1 x d], data_j: [1 x n x d]
    af::array data_i = af::moddims(data, n, 1, d);
    af::array data_j = af::moddims(data, 1, n, d);

    // Tile for broadcasting
    data_i = af::tile(data_i, 1, n, 1);
    data_j = af::tile(data_j, n, 1, 1);

    // Manhattan distance: sum(|a - b|)
    af::array diff = af::abs(data_i - data_j);
    return af::sum(diff, 2);  // [n x n]
}

af::array Clustering::ComputeCosineDistanceMatrix(const af::array& data) {
    int n = static_cast<int>(data.dims(0));
    (void)n;  // Suppress unused variable warning

    // Normalize data
    af::array norms = af::sqrt(af::sum(data * data, 1));  // [n x 1]
    norms = af::max(norms, 1e-10);  // Avoid division by zero
    af::array normalized = data / af::tile(norms, 1, static_cast<int>(data.dims(1)));

    // Cosine similarity = normalized @ normalized.T
    af::array similarity = af::matmul(normalized, normalized.T());

    // Cosine distance = 1 - similarity
    return 1.0 - similarity;
}

af::array Clustering::ComputeDistanceMatrix(const af::array& data, const std::string& metric) {
    if (metric == "manhattan") return ComputeManhattanDistanceMatrix(data);
    if (metric == "cosine") return ComputeCosineDistanceMatrix(data);
    return ComputeEuclideanDistanceMatrix(data);
}

af::array Clustering::ComputePointToCentroidDistances(const af::array& data, const af::array& centroids) {
    // data: [n_samples x n_features]
    // centroids: [n_clusters x n_features]
    // Output: [n_samples x n_clusters]

    int n = static_cast<int>(data.dims(0));
    int k = static_cast<int>(centroids.dims(0));

    // ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x.c
    af::array data_sq = af::sum(data * data, 1);          // [n x 1]
    af::array cent_sq = af::sum(centroids * centroids, 1); // [k x 1]

    af::array data_sq_tile = af::tile(data_sq, 1, k);      // [n x k]
    af::array cent_sq_tile = af::tile(cent_sq.T(), n, 1);  // [n x k]

    af::array dot = af::matmul(data, centroids.T());       // [n x k]

    af::array sq_distances = data_sq_tile + cent_sq_tile - 2.0 * dot;
    sq_distances = af::max(sq_distances, 0.0);

    return af::sqrt(sq_distances);
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
        spdlog::error("GMM failed: {}", result.error_message);
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
    af::array global_cov = af::matmul(data_centered.T(), data_centered) / static_cast<double>(n_samples - 1);

    for (int k = 0; k < n_components; ++k) {
        if (covariance_type == "spherical") {
            double var = af::mean<double>(af::diag(global_cov));
            covariances.push_back(af::identity(n_features, n_features, f64) * var);
        } else if (covariance_type == "diag") {
            covariances.push_back(af::diag(af::diag(global_cov), 0, false));
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
        log_probs(af::span, k) = af::log(pdf + 1e-300) + std::log(weights(k).scalar<double>());
    }

    // Log-sum-exp for numerical stability
    af::array max_log = af::max(log_probs, 1);
    af::array log_probs_shifted = log_probs - af::tile(max_log, 1, n_components);
    af::array sum_exp = af::sum(af::exp(log_probs_shifted), 1);
    af::array log_sum = max_log + af::log(sum_exp);

    // Responsibilities
    af::array responsibilities = af::exp(log_probs - af::tile(log_sum, 1, n_components));

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
    weights = nk / static_cast<double>(n_samples);

    // Update means
    for (int k = 0; k < n_components; ++k) {
        af::array resp_k = responsibilities(af::span, k);  // [n_samples x 1]
        af::array weighted_sum = af::sum(data * af::tile(resp_k, 1, n_features), 0);
        means(k, af::span) = weighted_sum / nk(k).scalar<double>();
    }

    // Update covariances
    for (int k = 0; k < n_components; ++k) {
        af::array resp_k = responsibilities(af::span, k);
        af::array mean_k = means(k, af::span);
        af::array diff = data - af::tile(mean_k, n_samples, 1);

        if (covariance_type == "spherical") {
            af::array weighted_sq = af::sum(diff * diff * af::tile(resp_k, 1, n_features), 0);
            double var = af::sum<double>(weighted_sq) / (nk(k).scalar<double>() * n_features);
            covariances[k] = af::identity(n_features, n_features, f64) * var;
        } else if (covariance_type == "diag") {
            af::array weighted_sq = af::sum(diff * diff * af::tile(resp_k, 1, n_features), 0);
            af::array variances = weighted_sq / nk(k).scalar<double>();
            covariances[k] = af::diag(variances.T(), 0, false);
        } else {
            af::array weighted_diff = diff * af::tile(af::sqrt(resp_k), 1, n_features);
            covariances[k] = af::matmul(weighted_diff.T(), weighted_diff) / nk(k).scalar<double>();
            // Add regularization
            covariances[k] += af::identity(n_features, n_features, f64) * 1e-6;
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

    // Compute (x - mu)^T @ Sigma^-1 @ (x - mu) for each sample
    af::array cov_inv = af::inverse(covariance);
    af::array mahalanobis = af::sum(af::matmul(diff, cov_inv) * diff, 1);

    double log_det = af::sum<double>(af::log(af::abs(af::diag(covariance))));
    double log_norm = -0.5 * (n_features * std::log(2 * M_PI) + log_det);

    return af::exp(log_norm - 0.5 * mahalanobis);
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
        weighted_probs += weights(k).scalar<double>() * pdf;
    }

    return af::sum<double>(af::log(weighted_probs + 1e-300));
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
