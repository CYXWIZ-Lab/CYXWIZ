#define _USE_MATH_DEFINES
#include "dimensionality_reduction.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <spdlog/spdlog.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz {

// ============================================================================
// Utility Methods
// ============================================================================

std::vector<double> DimensionalityReduction::MatVecMul(
    const std::vector<std::vector<double>>& matrix,
    const std::vector<double>& vec)
{
    size_t rows = matrix.size();
    size_t cols = vec.size();
    std::vector<double> result(rows, 0.0);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

double DimensionalityReduction::DotProduct(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

double DimensionalityReduction::Norm(const std::vector<double>& vec) {
    return std::sqrt(DotProduct(vec, vec));
}

void DimensionalityReduction::Normalize(std::vector<double>& vec) {
    double n = Norm(vec);
    if (n > 1e-10) {
        for (auto& v : vec) {
            v /= n;
        }
    }
}

// ============================================================================
// PCA Implementation
// ============================================================================

std::vector<std::vector<double>> DimensionalityReduction::CenterData(
    const std::vector<std::vector<double>>& data)
{
    if (data.empty()) return {};

    size_t n_samples = data.size();
    size_t n_features = data[0].size();

    // Compute column means
    std::vector<double> means(n_features, 0.0);
    for (const auto& row : data) {
        for (size_t j = 0; j < n_features; j++) {
            means[j] += row[j];
        }
    }
    for (auto& m : means) {
        m /= static_cast<double>(n_samples);
    }

    // Center data
    std::vector<std::vector<double>> centered(n_samples, std::vector<double>(n_features));
    for (size_t i = 0; i < n_samples; i++) {
        for (size_t j = 0; j < n_features; j++) {
            centered[i][j] = data[i][j] - means[j];
        }
    }

    return centered;
}

std::vector<std::vector<double>> DimensionalityReduction::ScaleData(
    const std::vector<std::vector<double>>& data)
{
    if (data.empty()) return {};

    size_t n_samples = data.size();
    size_t n_features = data[0].size();

    // Compute column standard deviations
    std::vector<double> stds(n_features, 0.0);
    for (const auto& row : data) {
        for (size_t j = 0; j < n_features; j++) {
            stds[j] += row[j] * row[j];
        }
    }
    for (auto& s : stds) {
        s = std::sqrt(s / static_cast<double>(n_samples));
        if (s < 1e-10) s = 1.0;  // Avoid division by zero
    }

    // Scale data
    std::vector<std::vector<double>> scaled(n_samples, std::vector<double>(n_features));
    for (size_t i = 0; i < n_samples; i++) {
        for (size_t j = 0; j < n_features; j++) {
            scaled[i][j] = data[i][j] / stds[j];
        }
    }

    return scaled;
}

std::vector<std::vector<double>> DimensionalityReduction::ComputeCovarianceMatrix(
    const std::vector<std::vector<double>>& data)
{
    if (data.empty()) return {};

    size_t n_samples = data.size();
    size_t n_features = data[0].size();

    // Initialize covariance matrix
    std::vector<std::vector<double>> cov(n_features, std::vector<double>(n_features, 0.0));

    // Compute C = X^T * X / (n-1)
    for (size_t i = 0; i < n_features; i++) {
        for (size_t j = i; j < n_features; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n_samples; k++) {
                sum += data[k][i] * data[k][j];
            }
            cov[i][j] = sum / static_cast<double>(n_samples - 1);
            cov[j][i] = cov[i][j];  // Symmetric
        }
    }

    return cov;
}

std::vector<double> DimensionalityReduction::PowerIteration(
    const std::vector<std::vector<double>>& matrix,
    int max_iterations,
    double tolerance)
{
    size_t n = matrix.size();
    std::vector<double> vec(n);

    // Initialize with random values
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto& v : vec) {
        v = dist(rng);
    }
    Normalize(vec);

    for (int iter = 0; iter < max_iterations; iter++) {
        std::vector<double> new_vec = MatVecMul(matrix, vec);
        Normalize(new_vec);

        // Check convergence
        double diff = 0.0;
        for (size_t i = 0; i < n; i++) {
            diff += std::abs(new_vec[i] - vec[i]);
        }

        vec = new_vec;

        if (diff < tolerance) {
            break;
        }
    }

    return vec;
}

void DimensionalityReduction::DeflateMatrix(
    std::vector<std::vector<double>>& matrix,
    const std::vector<double>& eigenvector,
    double eigenvalue)
{
    size_t n = matrix.size();
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
        }
    }
}

PCAResult DimensionalityReduction::ComputePCA(
    const std::vector<std::vector<double>>& data,
    int n_components,
    bool center,
    bool scale)
{
    PCAResult result;

    if (data.empty() || data[0].empty()) {
        result.success = false;
        result.error_message = "Empty input data";
        return result;
    }

    size_t n_samples = data.size();
    size_t n_features = data[0].size();

    if (n_components <= 0 || n_components > static_cast<int>(n_features)) {
        n_components = std::min(static_cast<int>(n_features), static_cast<int>(n_samples));
    }

    spdlog::info("Computing PCA: {} samples, {} features, {} components",
                 n_samples, n_features, n_components);

    // Preprocess data
    std::vector<std::vector<double>> processed = data;
    if (center) {
        processed = CenterData(processed);
    }
    if (scale) {
        processed = ScaleData(processed);
    }

    // Compute covariance matrix
    std::vector<std::vector<double>> cov = ComputeCovarianceMatrix(processed);

    // Extract eigenvalues and eigenvectors using power iteration
    result.components.resize(n_components);
    result.explained_variance.resize(n_components);
    result.singular_values.resize(n_components);

    std::vector<std::vector<double>> cov_copy = cov;
    double total_variance = 0.0;

    // First pass to get total variance (trace of covariance)
    for (size_t i = 0; i < n_features; i++) {
        total_variance += cov[i][i];
    }

    for (int comp = 0; comp < n_components; comp++) {
        // Power iteration to find dominant eigenvector
        std::vector<double> eigenvector = PowerIteration(cov_copy);

        // Compute eigenvalue (Rayleigh quotient)
        std::vector<double> Av = MatVecMul(cov_copy, eigenvector);
        double eigenvalue = DotProduct(eigenvector, Av);

        result.components[comp] = eigenvector;
        result.explained_variance[comp] = eigenvalue;
        result.singular_values[comp] = std::sqrt(eigenvalue * (n_samples - 1));

        // Deflate matrix
        DeflateMatrix(cov_copy, eigenvector, eigenvalue);
    }

    // Compute explained variance ratio
    result.explained_variance_ratio.resize(n_components);
    result.total_variance_explained = 0.0;
    for (int i = 0; i < n_components; i++) {
        result.explained_variance_ratio[i] = result.explained_variance[i] / total_variance;
        result.total_variance_explained += result.explained_variance_ratio[i];
    }

    // Project data onto principal components
    result.transformed.resize(n_samples, std::vector<double>(n_components));
    for (size_t i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_components; j++) {
            double proj = 0.0;
            for (size_t k = 0; k < n_features; k++) {
                proj += processed[i][k] * result.components[j][k];
            }
            result.transformed[i][j] = proj;
        }
    }

    result.n_components = n_components;
    result.success = true;

    spdlog::info("PCA complete: {:.1f}% variance explained",
                 result.total_variance_explained * 100);

    return result;
}

// ============================================================================
// t-SNE Implementation
// ============================================================================

std::vector<std::vector<double>> DimensionalityReduction::ComputeSquaredDistances(
    const std::vector<std::vector<double>>& data)
{
    size_t n = data.size();
    std::vector<std::vector<double>> distances(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            double dist_sq = 0.0;
            for (size_t k = 0; k < data[i].size(); k++) {
                double diff = data[i][k] - data[j][k];
                dist_sq += diff * diff;
            }
            distances[i][j] = dist_sq;
            distances[j][i] = dist_sq;
        }
    }

    return distances;
}

double DimensionalityReduction::FindSigma(
    const std::vector<double>& distances_i,
    int i,
    double target_entropy,
    double tolerance,
    int max_iterations)
{
    double sigma_min = 1e-20;
    double sigma_max = 1e10;
    double sigma = 1.0;

    for (int iter = 0; iter < max_iterations; iter++) {
        // Compute probabilities with current sigma
        double sum_p = 0.0;
        std::vector<double> probs(distances_i.size());

        for (size_t j = 0; j < distances_i.size(); j++) {
            if (static_cast<int>(j) != i) {
                probs[j] = std::exp(-distances_i[j] / (2.0 * sigma * sigma));
                sum_p += probs[j];
            }
        }

        if (sum_p < 1e-10) sum_p = 1e-10;

        // Compute entropy
        double entropy = 0.0;
        for (size_t j = 0; j < distances_i.size(); j++) {
            if (static_cast<int>(j) != i && probs[j] > 1e-10) {
                double p = probs[j] / sum_p;
                entropy -= p * std::log2(p);
            }
        }

        double diff = entropy - target_entropy;

        if (std::abs(diff) < tolerance) {
            break;
        }

        if (diff > 0) {
            // Entropy too high, decrease sigma
            sigma_max = sigma;
            sigma = (sigma + sigma_min) / 2.0;
        } else {
            // Entropy too low, increase sigma
            sigma_min = sigma;
            sigma = (sigma + sigma_max) / 2.0;
        }
    }

    return sigma;
}

std::vector<std::vector<double>> DimensionalityReduction::ComputeConditionalProbabilities(
    const std::vector<std::vector<double>>& squared_distances,
    int perplexity)
{
    size_t n = squared_distances.size();
    double target_entropy = std::log2(static_cast<double>(perplexity));

    std::vector<std::vector<double>> cond_probs(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; i++) {
        // Find sigma for this point
        double sigma = FindSigma(squared_distances[i], static_cast<int>(i), target_entropy);

        // Compute conditional probabilities
        double sum_p = 0.0;
        for (size_t j = 0; j < n; j++) {
            if (j != i) {
                cond_probs[i][j] = std::exp(-squared_distances[i][j] / (2.0 * sigma * sigma));
                sum_p += cond_probs[i][j];
            }
        }

        // Normalize
        if (sum_p > 1e-10) {
            for (size_t j = 0; j < n; j++) {
                cond_probs[i][j] /= sum_p;
            }
        }
    }

    return cond_probs;
}

std::vector<std::vector<double>> DimensionalityReduction::ComputeJointProbabilities(
    const std::vector<std::vector<double>>& cond_probs)
{
    size_t n = cond_probs.size();
    std::vector<std::vector<double>> joint(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            double p_ij = (cond_probs[i][j] + cond_probs[j][i]) / (2.0 * n);
            p_ij = std::max(p_ij, 1e-12);  // Avoid log(0)
            joint[i][j] = p_ij;
            joint[j][i] = p_ij;
        }
    }

    return joint;
}

std::vector<std::vector<double>> DimensionalityReduction::ComputeQDistribution(
    const std::vector<std::vector<double>>& embeddings,
    double& sum_q)
{
    size_t n = embeddings.size();
    std::vector<std::vector<double>> Q(n, std::vector<double>(n, 0.0));
    sum_q = 0.0;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            double dist_sq = 0.0;
            for (size_t k = 0; k < embeddings[i].size(); k++) {
                double diff = embeddings[i][k] - embeddings[j][k];
                dist_sq += diff * diff;
            }

            // Student's t-distribution with df=1: (1 + ||y_i - y_j||^2)^(-1)
            double q_ij = 1.0 / (1.0 + dist_sq);
            Q[i][j] = q_ij;
            Q[j][i] = q_ij;
            sum_q += 2.0 * q_ij;
        }
    }

    return Q;
}

std::vector<std::vector<double>> DimensionalityReduction::ComputetSNEGradient(
    const std::vector<std::vector<double>>& P,
    const std::vector<std::vector<double>>& Q,
    const std::vector<std::vector<double>>& embeddings,
    double sum_q)
{
    size_t n = embeddings.size();
    size_t d = embeddings[0].size();

    std::vector<std::vector<double>> gradient(n, std::vector<double>(d, 0.0));

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            if (i != j) {
                double q_ij = Q[i][j] / sum_q;
                double multiplier = 4.0 * (P[i][j] - q_ij) * Q[i][j];

                for (size_t k = 0; k < d; k++) {
                    gradient[i][k] += multiplier * (embeddings[i][k] - embeddings[j][k]);
                }
            }
        }
    }

    return gradient;
}

double DimensionalityReduction::ComputeKLDivergence(
    const std::vector<std::vector<double>>& P,
    const std::vector<std::vector<double>>& Q)
{
    double kl = 0.0;
    double sum_q = 0.0;

    // First compute sum_q
    for (size_t i = 0; i < Q.size(); i++) {
        for (size_t j = i + 1; j < Q.size(); j++) {
            sum_q += 2.0 * Q[i][j];
        }
    }

    // Then compute KL divergence
    for (size_t i = 0; i < P.size(); i++) {
        for (size_t j = i + 1; j < P.size(); j++) {
            if (P[i][j] > 1e-12 && Q[i][j] > 1e-12) {
                double q_ij = Q[i][j] / sum_q;
                kl += P[i][j] * std::log(P[i][j] / q_ij);
            }
        }
    }

    return 2.0 * kl;  // Symmetrize
}

tSNEResult DimensionalityReduction::ComputetSNE(
    const std::vector<std::vector<double>>& data,
    int n_dims,
    int perplexity,
    double learning_rate,
    int n_iterations,
    std::function<void(int, double)> progress_callback)
{
    tSNEResult result;

    if (data.empty() || data[0].empty()) {
        result.success = false;
        result.error_message = "Empty input data";
        return result;
    }

    size_t n_samples = data.size();

    // Adjust perplexity if needed
    perplexity = std::min(perplexity, static_cast<int>(n_samples) / 3);
    perplexity = std::max(perplexity, 5);

    spdlog::info("Computing t-SNE: {} samples, {} dims, perplexity={}",
                 n_samples, n_dims, perplexity);

    // Step 1: Compute pairwise distances in high-D space
    std::vector<std::vector<double>> squared_distances = ComputeSquaredDistances(data);

    // Step 2: Compute conditional probabilities
    std::vector<std::vector<double>> cond_probs = ComputeConditionalProbabilities(
        squared_distances, perplexity);

    // Step 3: Compute joint probabilities P
    std::vector<std::vector<double>> P = ComputeJointProbabilities(cond_probs);

    // Early exaggeration (multiply P by 4 for first 100 iterations)
    for (size_t i = 0; i < n_samples; i++) {
        for (size_t j = 0; j < n_samples; j++) {
            P[i][j] *= 4.0;
        }
    }

    // Step 4: Initialize low-D embeddings randomly
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1e-4);

    result.embeddings.resize(n_samples, std::vector<double>(n_dims));
    for (size_t i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_dims; j++) {
            result.embeddings[i][j] = dist(rng);
        }
    }

    // Momentum
    std::vector<std::vector<double>> velocity(n_samples, std::vector<double>(n_dims, 0.0));
    double momentum = 0.5;

    // Step 5: Gradient descent
    result.kl_divergence_history.reserve(n_iterations / 10);

    for (int iter = 0; iter < n_iterations; iter++) {
        // Remove early exaggeration after 100 iterations
        if (iter == 100) {
            for (size_t i = 0; i < n_samples; i++) {
                for (size_t j = 0; j < n_samples; j++) {
                    P[i][j] /= 4.0;
                }
            }
            momentum = 0.8;
        }

        // Compute Q distribution
        double sum_q = 0.0;
        std::vector<std::vector<double>> Q = ComputeQDistribution(result.embeddings, sum_q);

        // Compute gradient
        std::vector<std::vector<double>> gradient = ComputetSNEGradient(
            P, Q, result.embeddings, sum_q);

        // Update embeddings with momentum
        for (size_t i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_dims; j++) {
                velocity[i][j] = momentum * velocity[i][j] - learning_rate * gradient[i][j];
                result.embeddings[i][j] += velocity[i][j];
            }
        }

        // Track KL divergence periodically
        if (iter % 50 == 0 || iter == n_iterations - 1) {
            double kl = ComputeKLDivergence(P, Q);
            result.kl_divergence_history.push_back(kl);
            result.final_kl_divergence = kl;

            if (progress_callback) {
                progress_callback(iter, kl);
            }

            spdlog::debug("t-SNE iter {}: KL = {:.4f}", iter, kl);
        }
    }

    result.n_iterations = n_iterations;
    result.perplexity = perplexity;
    result.success = true;

    spdlog::info("t-SNE complete: final KL = {:.4f}", result.final_kl_divergence);

    return result;
}

// ============================================================================
// UMAP Implementation (Simplified)
// ============================================================================

std::vector<std::vector<int>> DimensionalityReduction::FindKNearestNeighbors(
    const std::vector<std::vector<double>>& squared_distances,
    int k)
{
    size_t n = squared_distances.size();
    std::vector<std::vector<int>> neighbors(n);

    for (size_t i = 0; i < n; i++) {
        // Create pairs of (distance, index)
        std::vector<std::pair<double, int>> dist_idx;
        for (size_t j = 0; j < n; j++) {
            if (j != i) {
                dist_idx.push_back({squared_distances[i][j], static_cast<int>(j)});
            }
        }

        // Sort by distance
        std::sort(dist_idx.begin(), dist_idx.end());

        // Take k nearest
        neighbors[i].resize(k);
        for (int j = 0; j < k && j < static_cast<int>(dist_idx.size()); j++) {
            neighbors[i][j] = dist_idx[j].second;
        }
    }

    return neighbors;
}

std::vector<std::vector<double>> DimensionalityReduction::ComputeUMAPWeights(
    const std::vector<std::vector<double>>& squared_distances,
    const std::vector<std::vector<int>>& neighbors,
    int n_neighbors)
{
    size_t n = squared_distances.size();
    std::vector<std::vector<double>> weights(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; i++) {
        // Find local connectivity (distance to nearest neighbor)
        double rho = std::sqrt(squared_distances[i][neighbors[i][0]]);

        // Find sigma using binary search
        double sigma = 1.0;
        double target = std::log2(static_cast<double>(n_neighbors));

        for (int iter = 0; iter < 64; iter++) {
            double sum = 0.0;
            for (int j = 0; j < n_neighbors; j++) {
                double dist = std::sqrt(squared_distances[i][neighbors[i][j]]);
                sum += std::exp(-(std::max(0.0, dist - rho)) / sigma);
            }

            double log_sum = std::log2(std::max(sum, 1e-10));
            if (std::abs(log_sum - target) < 0.01) break;

            if (log_sum > target) sigma *= 0.5;
            else sigma *= 2.0;
        }

        // Compute weights for neighbors
        for (int j = 0; j < n_neighbors; j++) {
            int idx = neighbors[i][j];
            double dist = std::sqrt(squared_distances[i][idx]);
            weights[i][idx] = std::exp(-(std::max(0.0, dist - rho)) / sigma);
        }
    }

    // Symmetrize: w_ij = w_ij + w_ji - w_ij * w_ji
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            double w = weights[i][j] + weights[j][i] - weights[i][j] * weights[j][i];
            weights[i][j] = w;
            weights[j][i] = w;
        }
    }

    return weights;
}

UMAPResult DimensionalityReduction::ComputeUMAP(
    const std::vector<std::vector<double>>& data,
    int n_dims,
    int n_neighbors,
    double min_dist)
{
    UMAPResult result;

    if (data.empty() || data[0].empty()) {
        result.success = false;
        result.error_message = "Empty input data";
        return result;
    }

    size_t n_samples = data.size();
    n_neighbors = std::min(n_neighbors, static_cast<int>(n_samples) - 1);

    spdlog::info("Computing UMAP: {} samples, {} dims, {} neighbors",
                 n_samples, n_dims, n_neighbors);

    // Compute distances
    std::vector<std::vector<double>> squared_distances = ComputeSquaredDistances(data);

    // Find k-nearest neighbors
    std::vector<std::vector<int>> neighbors = FindKNearestNeighbors(squared_distances, n_neighbors);

    // Compute UMAP graph weights
    std::vector<std::vector<double>> weights = ComputeUMAPWeights(
        squared_distances, neighbors, n_neighbors);

    // Initialize embeddings using spectral initialization or random
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    result.embeddings.resize(n_samples, std::vector<double>(n_dims));
    for (size_t i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_dims; j++) {
            result.embeddings[i][j] = dist(rng);
        }
    }

    // UMAP parameters
    double a = 1.0, b = 1.0;

    // Compute a and b from min_dist (simplified)
    if (min_dist < 1.0) {
        b = 1.0;
        a = 1.577 * std::pow(min_dist, 0.8951);
    }

    // Optimization (simplified SGD)
    double learning_rate = 1.0;
    int n_epochs = 200;

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        learning_rate = 1.0 * (1.0 - static_cast<double>(epoch) / n_epochs);

        for (size_t i = 0; i < n_samples; i++) {
            // Attractive forces from neighbors
            for (int k = 0; k < n_neighbors; k++) {
                int j = neighbors[i][k];
                double w = weights[i][j];
                if (w < 1e-10) continue;

                double dist_sq = 0.0;
                for (int d = 0; d < n_dims; d++) {
                    double diff = result.embeddings[i][d] - result.embeddings[j][d];
                    dist_sq += diff * diff;
                }

                double grad_coeff = -2.0 * a * b * std::pow(dist_sq, b - 1.0) /
                                   (1.0 + a * std::pow(dist_sq, b));

                for (int d = 0; d < n_dims; d++) {
                    double diff = result.embeddings[i][d] - result.embeddings[j][d];
                    result.embeddings[i][d] -= learning_rate * w * grad_coeff * diff;
                }
            }

            // Repulsive forces (sample negative examples)
            for (int neg = 0; neg < 5; neg++) {
                int j = rng() % n_samples;
                if (j == static_cast<int>(i)) continue;

                double dist_sq = 0.0;
                for (int d = 0; d < n_dims; d++) {
                    double diff = result.embeddings[i][d] - result.embeddings[j][d];
                    dist_sq += diff * diff;
                }

                double grad_coeff = 2.0 * b / ((0.001 + dist_sq) * (1.0 + a * std::pow(dist_sq, b)));

                for (int d = 0; d < n_dims; d++) {
                    double diff = result.embeddings[i][d] - result.embeddings[j][d];
                    result.embeddings[i][d] -= learning_rate * (1.0 - weights[i][j]) * grad_coeff * diff;
                }
            }
        }
    }

    result.n_neighbors = n_neighbors;
    result.min_dist = min_dist;
    result.success = true;

    spdlog::info("UMAP complete");

    return result;
}

} // namespace cyxwiz
